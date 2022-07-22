import argparse
import copy
import dataclasses
import json
import logging
import os
import pathlib
import xml.dom.minidom
from enum import Enum, auto
from json import JSONDecodeError
from multiprocessing import Pool
from os import listdir
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Union, Callable
from xml.dom.minidom import Document

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.data import AUTOTUNE
from tqdm import tqdm

from config import BATCH_SIZE_DEFAULT, MAX_WORD_LEN_DEFAULT, WHITELIST_PATH_DEFAULT
from errors import Error
from utils.common import Rect, GroundTruthPathsItem, Word, Line, ROI, Dataset, GroundTruth, Point
from utils.image_utils import tf_distortion_free_resize, augment_image, distortion_free_resize, \
    magnie_humie, load_and_pad_image, AUGThresholdMode
from utils.text_utils import Vocabulary

PUNCTUATIONS = {",", ".", '?', "!", ":", ";"}


GROUND_TRUTH_FILENAME = "ground_truth.txt"

IMG_NAME_IDX = 0
VALUE_IDX = 8

LOG = logging.getLogger(__name__)


class GTFormat(Enum):
    IAM_ASCII = auto()
    IAM_XML = auto()
    PAGE_XML = auto()
    JSON = auto()
    TESSERACT = auto()


def get_img_locs(img_dir: str) -> Dict[str, str]:
    res = {}
    _get_img_locs_recursive(img_dir, res)
    return res


def _get_img_locs_recursive(
    parent: str,
    res: Dict[str, str]
) -> Dict[str, str]:
    dir_items: List[str] = listdir(parent)
    for di in dir_items:
        di_full = os.path.join(parent, di)
        if os.path.isdir(di_full):
            _get_img_locs_recursive(di_full, res)
        else:
            di_noext = di.split(".")[0]
            res[di_noext] = di_full

    return res


def make_lines_dataset(
    lines_root: pathlib.Path,
    forms_xml_root: pathlib.Path,
    forms_img_root: pathlib.Path,
    page_xml: bool,
    filter: Callable[[GroundTruthPathsItem], bool] = None,
    threshold: bool = False,
) -> Dataset:
    """
    Loads forms information (lines, words), and renders it into lines
    skipping blacklisted words
    :param lines_root: directory where rendered lines will be saved
    :param forms_xml_root: directory which holds form XML files
    :param forms_img_root: directory which holds form image files
    :param threshold: if set, then threshold will be applied to each word
    :param page_xml: use PAGE-XML instead of IAM XML
    :param filter: callback if provided should return False for words to skip
    :return: dataset with lines
    """

    def _get_page_xml_word_text(w_node):
        text_equiv = w_node.getElementsByTagName("TextEquiv").item(0)
        unicode_node = text_equiv.getElementsByTagName("Unicode").item(0)
        return unicode_node.firstChild.data

    def _get_page_xml_line_id(form_name, line_idx):
        return f"{form_name}-line_{line_idx}"

    def _get_page_xml_word_id(form_name, line_idx, word_idx):
        return f"{form_name}-line_{line_idx}-word_{word_idx}"

    def _get_lines_page_xml(form_path: pathlib.Path):
        dom: Document = xml.dom.minidom.parse(str(form_path))
        pc_gts_node = dom.getElementsByTagName("PcGts").item(0)
        page = pc_gts_node.getElementsByTagName("Page").item(0)
        words = page.getElementsByTagName("Word")
        form_name = form_path.stem

        lines = dict()
        cur_line = []
        prev_word_left = -1
        for w_dom in words:
            coords_node = w_dom.getElementsByTagName("Coords").item(0)
            points = coords_node.getAttribute("points").strip().split(" ")
            lt, rt, rb, lb = tuple([
                tuple(map(int, pt.split(",")))
                for pt in points
            ])
            left, top = lt
            right, bottom = rb

            def _mk_word(word_idx):
                return Word(
                    word_id=_get_page_xml_word_id(form_name, len(lines), word_idx),
                    text=_get_page_xml_word_text(w_dom),
                    glyphs=[Rect(left, top, right-left, bottom-top)]
                )

            if prev_word_left < right:
                cur_line.append(_mk_word(len(cur_line)))
            else:
                lines[_get_page_xml_line_id(form_name, len(lines))] = Line(
                    text=" ".join(map(lambda lnw: lnw.text, cur_line)),
                    words=cur_line
                )
                cur_line = [_mk_word(0)]
            prev_word_left, _ = lt
        return lines

    def _get_words(line_node: xml.dom.Node) -> List[Word]:
        words = [
            Word(
                node.getAttribute("id"),
                node.getAttribute("text"),
                [
                    Rect(
                        int(rect_node.getAttribute("x")),
                        int(rect_node.getAttribute("y")),
                        int(rect_node.getAttribute("width")),
                        int(rect_node.getAttribute("height")),
                    )
                    for rect_node in node.getElementsByTagName("cmp")
                ]
            )
            for node in line_node.getElementsByTagName("word")
        ]
        return words

    def _get_lines(form_path: pathlib.Path) -> Dict[str, Line]:
        dom: Document = xml.dom.minidom.parse(str(form_path))
        lines_dom = {
            node.getAttribute("id"): Line(
                node.getAttribute("text"),
                _get_words(node)
            )
            for node in dom.getElementsByTagName("line")
        }
        return lines_dom

    def _render_rois(
        dest_img_path: Path,
        form_img: np.ndarray,
        rois_new_old: List[Tuple[Point, Rect]]
    ):
        new_positions, rois_old = list(zip(*rois_new_old))

        left = new_positions[0].x
        top = min(new_positions, key=lambda pt: pt.y).y

        right_pt = max(rois_new_old, key=lambda pt: pt[0].x + pt[1].width)
        right = right_pt[0].x + right_pt[1].width

        bottom_pt = max(rois_new_old, key=lambda pt: pt[0].y + pt[1].height)
        bottom = bottom_pt[0].y + bottom_pt[1].height

        res = np.full([bottom - top, right - left], 255, np.uint8)

        for new_pos, roi_old in rois_new_old:
            new_x, new_y = new_pos.to_vec()
            old_x, old_y, width, height = roi_old.to_vec()

            w_img = form_img[old_y: old_y + height, old_x: old_x + width]

            if threshold:
                w_img = augment_image(w_img, AUGThresholdMode.ONLY_THRESHOLD)["threshold"]

            render_x = new_x - left
            render_y = new_y - top

            dest = res[
                render_y: render_y + height,
                render_x: render_x + width,
            ]

            assert w_img.shape == dest.shape

            res[
                render_y: render_y + height,
                render_x: render_x + width,
            ] = w_img

        cv2.imwrite(str(dest_img_path), res)

    res: Dataset = []

    image_extension = ".png" if not page_xml else ".jpg"

    dir_items: List[pathlib.Path] = list(map(pathlib.Path, listdir(forms_xml_root)))
    locs = {
        pathlib.Path(item): (forms_img_root / item.with_suffix(image_extension))
        for item in dir_items if item.suffix == ".xml"
    }

    total_skipped_words = 0
    total_skipped_lines = 0
    total_lines = 0

    for form_xml, form_img_path in tqdm(locs.items(), desc="Processing forms"):
        if not form_img_path.exists():
            continue
        form_img = load_and_pad_image(form_img_path, pad_resize=False)
        if form_img is None:
            continue
        form_img = form_img.reshape(form_img.shape[:-1])
        lines: Dict[str, Line] = _get_lines(forms_xml_root / form_xml) if not page_xml \
            else _get_lines_page_xml(forms_xml_root / form_xml)
        for ln_id, ln in lines.items():
            total_lines += 1
            num_skipped = 0
            first_skipped_before: Optional[Word] = None
            skipped_before: List[Word] = []
            strikeout_offset = 0
            str_value = []
            words_to_render: List[Tuple[Point, Rect]] = []
            for i, word in enumerate(ln.words):
                if not word.glyphs:
                    continue

                word_gt_item = GroundTruthPathsItem(
                    str_value=word.text,
                    img_name=word.word_id,
                    img_path=form_img_path,
                    roi=word.get_rect().to_vec()
                )

                passed = filter(word_gt_item)

                if not passed:
                    if first_skipped_before is None:
                        first_skipped_before = word
                    skipped_before.append(word)
                    total_skipped_words += 1
                    continue

                elif first_skipped_before is not None:
                    # We do some trick to keep overlapping words at proper spacing
                    # from one side,
                    # and to keep non-overlapped spaces as well. The latter is implicitly
                    # defined by differences between word boundaries.
                    max_skipped_x = max(skipped_before, key=lambda w: w.get_right()).get_right()
                    strikeout_offset += \
                        min(max_skipped_x, word.get_left()) \
                        - first_skipped_before.get_left()

                    first_skipped_before = None
                    skipped_before = []

                if words_to_render and word.text not in PUNCTUATIONS:
                    str_value.append(" ")

                old_rect = word.get_rect()
                new_x = old_rect.x - strikeout_offset
                assert new_x >= 0
                words_to_render.append((Point(new_x, old_rect.y), old_rect))
                str_value.append(word.text)

            if words_to_render:
                str_value = "".join(str_value)
                line_img_path = (lines_root / ln_id).with_suffix(image_extension)
                _render_rois(line_img_path, form_img, words_to_render)

                LOG.debug(f"Adding:")
                LOG.debug(f"    Value: {str_value}")
                LOG.debug(f"    Path: {form_img_path}")
                LOG.debug(f"    Num words skipped: {num_skipped}")
                res.append(GroundTruthPathsItem(
                    str_value=str_value,
                    img_name=ln_id,
                    img_path=str(line_img_path),
                ))
            else:
                total_skipped_lines += 1

    # It seems to duplicate message emitted by load_dataset
    #    LOG.info(f"    Total words skipped: {total_skipped_words}")
    LOG.info(f"    Total lines checked: {total_lines}")
    LOG.info(f"    Total lines skipped: {total_skipped_lines}")
    LOG.info(f"    Total lines rendered: {total_lines - total_skipped_lines}")

    return res


def dataset_to_ground_truth(ds: Dataset) -> GroundTruth:
    return {gt.img_path: gt for gt in ds}


def ground_truth_to_dataset(ground_truth: GroundTruth) -> Dataset:
    return list(sorted(ground_truth.values(), key=lambda gt: gt.img_name))


def load_ground_truth_json(
    path: Union[str, os.PathLike],
    on_sample: Callable[[GroundTruthPathsItem], bool] = None,
    max_gt_items: int = None
) -> GroundTruth:
    with open(path, "r") as f:
        try:
            dd: Dict = json.load(f)

            items = dd.items()
            if max_gt_items:
                items = list(items)
                items = items[:min(max_gt_items, len(items))]

            return {
                img_path: ggt
                for img_path, gt in items
                for ggt in [GroundTruthPathsItem(**gt)]
                if not on_sample or on_sample(ggt)
            }
        except JSONDecodeError as e:
            LOG.warning(f"Unable to decode '{path}': {e}")


def save_ground_truth_json(
    ground_truth: Dict[str, GroundTruthPathsItem],
    path: Union[str, os.PathLike],
    gt_formats: Set[GTFormat] = None
):
    supported_formats = {GTFormat.JSON, GTFormat.TESSERACT}

    if not gt_formats:
        gt_formats = {GTFormat.JSON}

    if GTFormat.JSON in gt_formats:
        with open(path, "w") as f:
            d = {
                img_path: dataclasses.asdict(gt)
                for img_path, gt in ground_truth.items()
            }
            json.dump(d, f, indent=4)
    if GTFormat.TESSERACT in gt_formats:
        for img_path, gt in tqdm(ground_truth.items(), desc="Saving tesseract .gt.txt"):
            gt_path = Path(img_path).with_suffix(".gt.txt")
            with open(gt_path, "w") as f:
                print(gt.str_value, file=f)

    unsupported = gt_formats.difference(supported_formats)
    if unsupported:
        for gt_format in unsupported:
            LOG.warning(f"Can not save ground truth in '{gt_format}'")


def load_iam_dataset(
    str_values_file_path: str,
    img_dir: str,
    on_sample: Callable[[GroundTruthPathsItem], bool],
    max_ds_items: int
):
    res = []
    locs = get_img_locs(img_dir)

    with open(str_values_file_path, "r") as f:
        lines = [ll for ll in f.readlines() if not ll.startswith("#")]
        if max_ds_items:
            lines = lines[:min(len(lines), max_ds_items)]

        LOG.debug(f"Total lines/sentences read: {len(lines)}")

        for line in lines:
            l_items = line.split(" ")
            str_value = l_items[VALUE_IDX].strip().replace("|", " ")
            img_name = l_items[IMG_NAME_IDX]
            img_path = locs[img_name]

            sample = GroundTruthPathsItem(
                str_value=str_value,
                img_path=img_path,
                img_name=img_name
            )

            if not on_sample or on_sample(sample):
                res.append(sample)

    return res


@dataclasses.dataclass()
class WordsFilter:
    """
    Image names which are blacklisted
    """
    blacklist: Set[str] = None

    """
    Image names which are whitelisted
    """
    whitelist: Set[str] = None

    """
    Ground-truth values to be ignored
    """
    ignore_list: Set[str] = None

    """
    Characters which are allowed
    """
    allowed_characters: Set[str] = None

    max_word_len: int = 0
    max_ds_items: int = 0
    num_skipped: int = 0
    total_loaded: int = 0

    def __call__(self, gt: GroundTruthPathsItem):
        self.total_loaded += 1

        def log_skip(reason: str):
            LOG.debug(f"Skipping word '{gt.str_value}', rendered as '{gt.img_path}': {reason}")

        if self.whitelist and gt.img_name not in self.whitelist:
            log_skip("not in whitelist.")
            self.num_skipped += 1
            return False

        if self.max_word_len and len(gt.str_value) > self.max_word_len:
            log_skip(f"exceeds max length {self.max_word_len} characters.")
            self.num_skipped += 1
            return False

        if self.ignore_list and gt.str_value in self.ignore_list:
            log_skip("is in ignore list")
            self.num_skipped += 1
            return False

        if self.blacklist and gt.img_name in self.blacklist:
            log_skip("is in blacklist")
            self.num_skipped += 1
            return False

        if self.allowed_characters:
            disallowed = set(gt.str_value).difference(self.allowed_characters)
            if disallowed:
                log_skip(f"contains disallowed characters: {''.join(disallowed)}")
                self.num_skipped += 1
                return False

        if not Path(gt.img_path).exists() or os.path.getsize(gt.img_path) == 0:
            log_skip("image is empty or absent")
            self.num_skipped += 1
            return False

        return True


def load_dataset(
    gt_format: GTFormat,
    gt_path: str,
    img_dir: str,
    vocabulary: Optional[Vocabulary] = None,
    apply_ignore_list: bool = False,
    blacklist: Optional[Set[str]] = None,
    whitelist: Optional[Set[str]] = None,
    max_ds_items: int = 0,
) -> Tuple[Dataset, Vocabulary]:
    """
    Loads dataset in our own format.
    We intended to use such format because it is easier to evaluate and check.
    We could use internal tensorflow format, it it might be stored out
    of python context (in native part). In this case it might be hard to check
    its contents during debug.
    :param gt_path: path to ground truth values (IAM ASCII format)
    :param img_dir: root path to images directory.
    :param vocabulary: vocabulary if provided, then it will be used to filter
       words which are too long, or which use inappropriate characters
    :param apply_ignore_list: apply ignore list from vocabulary
    :param blacklist: set of blacklisted dataset items
    :param whitelist: set of explicitly whitelisted dataset items
    :param max_ds_items: dataset size limit
    :return: Dataset instance (which is a list)
    """
    max_word_len = vocabulary.max_len if vocabulary else 0
    auto_voc = Vocabulary()

    ignore_list = set(vocabulary.ignore) if vocabulary and apply_ignore_list else None

    allowed_characters = set(vocabulary.characters) if vocabulary and apply_ignore_list else None

    filter = WordsFilter(
        blacklist=blacklist, whitelist=whitelist,
        ignore_list=ignore_list,
        allowed_characters=allowed_characters,
        max_word_len=max_word_len,
        max_ds_items=max_ds_items
    )

    with auto_voc.builder() as characters:
        if gt_format == GTFormat.JSON:
            ground_truth = load_ground_truth_json(gt_path, filter, max_ds_items)
            res = ground_truth_to_dataset(ground_truth), auto_voc
        elif gt_format == GTFormat.IAM_ASCII:
            res = load_iam_dataset(gt_path, img_dir, filter, max_ds_items), auto_voc
        # elif gt_format == GTFormat.IAM_XML:
        #     res = load_iam_xml(gt_path, img_dir, on_sample, max_ds_items), auto_voc
        else:
            raise Error(f"Ground truth is in wrong format.")
        pass

    if not res:
        raise Error("Final dataset is empty.")

    if whitelist:
        loaded_items = {gt.img_name for gt in res[0]}
        for w in whitelist:
            if w not in loaded_items:
                LOG.warning(f"Item '{w}' is in whitelist, but not loaded.")

    LOG.info(f"Total samples checked: {filter.total_loaded}")
    LOG.info(f"Final dataset size: {len(res[0])}")
    LOG.info(f"Total amount of skipped words: {filter.num_skipped}")
    return res


def _preprocess_item(
    src_item: GroundTruthPathsItem,
    cache_dir: str,
    only_threshold: bool = False,
    keep_existing_augmentations: bool = False,
    ignore_augmentations: Set[str] = None,
    resize=True,
    threshold=True,
    subdir=True,
) -> Optional[Tuple[GroundTruthPathsItem, Dataset]]:

    img_path = src_item.img_path
    roi = src_item.roi
    threshold_mode = AUGThresholdMode.ONLY_THRESHOLD if only_threshold else \
        AUGThresholdMode.NO_THRESHOLD if not threshold else \
        AUGThresholdMode.FULL

    cache_subdir = Path(cache_dir) / src_item.img_name if subdir else Path(cache_dir)
    ext = Path(img_path).suffix

    if keep_existing_augmentations and cache_subdir.exists():
        res = [
            GroundTruthPathsItem(
                str_value=src_item.str_value,
                img_name=f.stem,
                img_path=str(f),
                roi=None
            )
            for f in cache_subdir.iterdir() if f.is_file() and f.suffix in {".png", ".jpg"}
        ]
        if res:
            LOG.debug(f"Augmentation for '{src_item.img_name}' exists, skipping")
            return src_item, res
        # Otherwise directory is empty, fallthrough

    try:
        image: np.ndarray = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
        if image is None:
            LOG.warning(f"Unable to load '{img_path}'")
            return None
    except Exception as e:
        LOG.warning(f"Unable to load '{img_path}': {e}")
        return None

    if len(image.shape) == 2:
        image = image.reshape([*image.shape, 1])

    def save_augmentations(augs_dict):
        res = []
        for aug_name, aug in augs_dict.items():
            aug_name = f"{src_item.img_name}-{aug_name}" if aug_name else src_item.img_name
            dest_path = cache_subdir / f"{aug_name}{ext}"
            cv2.imwrite(str(dest_path), distortion_free_resize(aug) if resize else aug)
            res.append(GroundTruthPathsItem(
                str_value=src_item.str_value,
                img_name=aug_name,
                img_path=str(dest_path),
                roi=None
            ))
        return res

    if roi:
        left, top, width, height = roi
        image = image[top:top+height, left:left+width, :]

    cache_subdir.mkdir(parents=True, exist_ok=True)

    if only_threshold:
        assert threshold, "Thresholding must be enabled"
        thresh = augment_image(image, AUGThresholdMode.ONLY_THRESHOLD)["threshold"]
        res = save_augmentations({"": thresh})
        return src_item, res

    # Apply different sorts of threshold
    magnie, humie = magnie_humie(image)
    pre_aug = {
        "": image,
        "magnie": magnie,
        "humie": humie,
    }
    augmentation = {}
    for prefix, pre_aug_image in pre_aug.items():
        sub_aug = augment_image(pre_aug_image, threshold_mode)

        if not only_threshold and ignore_augmentations:
            for aug in ignore_augmentations:
                sub_aug.pop(aug)

        sub_aug = {
            (f"{prefix}-{name}" if prefix else name): augmented
            for name, augmented in sub_aug.items()
        }
        augmentation.update(sub_aug)

    res: Dataset = []

    res.extend(save_augmentations(augmentation))

    return src_item, res


def _preprocess_item_task(args_dict):
    return _preprocess_item(**args_dict)


def preprocess_dataset(
    ds: Dataset,
    only_threshold: bool,
    cache_dir: str,
    keep: bool = False,
    full: bool = False,
    resize: bool = True,
    threshold: bool = True,
    subdir: bool = True,
    jobs: int = 1,
) -> Dataset:
    """
    Runs image preprocessing and augmentation
    :param ds: source dataset
    :param only_threshold: don't create augmented images, only apply threshold
    :param cache_dir: cache directory where preprocessing results will be stored
    :param keep: use cached items instead of making new ones
    :param full: also include blurred and adaptive thresholded (and may be more)
    :param threshold: apply item thresholding. TODO: merge with only_threshold param
    :param resize: resize item do default model input size
    :param subdir: Put each set of preprocessed items into subdir"
        which is named after original item.
    :param jobs: Amount of parallel jobs.
    :return: modified dataset with paths targeting to cache directory
    """

    res: Dataset = []
    cache_dir = pathlib.Path(cache_dir)

    preprocess_args = [dict(
        src_item=d,
        cache_dir=cache_dir,
        only_threshold=only_threshold,
        keep_existing_augmentations=keep,
        ignore_augmentations={"blurred", "adaptive_threshold"} if not full else None,
        resize=resize,
        threshold=threshold,
        subdir=subdir,
    ) for d in ds]

    def apply_preprocess_result(d, gts):
        if full:
            res.append(d)
        res.extend(gts)

    if jobs > 1:
        LOG.info("Launching DS preprocessing pool...")
        pool = Pool(processes=jobs)

        for d, gts in tqdm(
            pool.imap_unordered(_preprocess_item_task, preprocess_args),
            total=len(ds),
            desc="Preprocessing dataset (async)"
        ):
            apply_preprocess_result(d, gts)
        res = list(sorted(res, key=lambda gt: gt.img_name))
    else:
        for args in tqdm(preprocess_args, desc="Preprocessing dataset"):
            d, gts = _preprocess_item(**args)
            apply_preprocess_result(d, gts)

    ground_truth_path = pathlib.Path(cache_dir) / "gt.json"

    stored_dataset = dataset_to_ground_truth(res)

    if ground_truth_path.exists():
        stored_dataset.update(load_ground_truth_json(ground_truth_path))

    save_ground_truth_json(stored_dataset, ground_truth_path)

    return res


def extract_images_and_labels(ds: Dataset) -> Tuple[List[str], List[str], List[ROI]]:
    paths = []
    labels = []
    rois = []

    for gt in ds:
        labels.append(gt.str_value)
        paths.append(gt.img_path)
        rois.append(gt.roi if gt.roi else (0, 0, 0, 0))

    return paths, labels, rois


def tf_dataset(
    ds: Dataset,
    vocabulary: Vocabulary,
    resize: bool = True,
    shuffle: bool = True,
    batch_size: int = BATCH_SIZE_DEFAULT,
) -> tf.data.Dataset:
    """
    Converts dataset to internal tensorflow representation
    :param ds:
    :param vocabulary: is used to vectorize labels properly
    :param resize: If set, then image will be resized.
    :param batch_size: Amount of samples per batch
    :return: tf.Data.Dataset instance
    """

    paths, labels, rois = extract_images_and_labels(ds)

    def _process_images_labels(img_path, label, roi):
        label = vocabulary.vectorize_label(label)

        img_bytes = tf.io.read_file(img_path)
        image = tf.image.decode_image(img_bytes, 1, expand_animations=False)

        if roi[2] != 0:
            image = tf.image.crop_to_bounding_box(image, roi[1], roi[0], roi[3], roi[2])

        if resize:
            image = tf_distortion_free_resize(image)
        else:
            image = tf.transpose(image, perm=[1, 0, 2])
            image = tf.image.flip_left_right(image)

        image = tf.cast(image, tf.float32) / 255.0

        return {"image": image, "label": label}

    tf_ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels, rois)
    ).map(_process_images_labels, num_parallel_calls=AUTOTUNE)

    if shuffle:
        return tf_ds.shuffle(len(ds)).batch(batch_size).prefetch(AUTOTUNE).cache()
    return tf_ds.batch(batch_size).prefetch(AUTOTUNE).cache()


@dataclasses.dataclass
class MarkedState:
    marked: List[str] = dataclasses.field(default_factory=list)
    current_page: int = 0
    start_item: str = ""


def load_marked(state_path: str) -> MarkedState:
    with open(state_path, "r") as ff:
        vv = json.load(ff)
        return MarkedState(**vv)


def save_marked(state_path: str, state: MarkedState):
    v = dataclasses.asdict(state)
    with open(state_path, "w") as f:
        json.dump(v, f, indent=4)


def save_whitelist(dataset):
    whitelisted = [gt.img_name for gt in dataset]
    marked = MarkedState(marked=whitelisted)
    LOG.info(f"Saving whitelist to {WHITELIST_PATH_DEFAULT}.")
    save_marked(WHITELIST_PATH_DEFAULT, marked)


def add_blacklist_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-blacklist",
        help="File with blacklisted item names. Should be in format of 'ploti' state."
    )
    parser.add_argument(
        "-whitelist",
        help="File with whitelist item names. Should be in format of 'ploti' state."
             " If specified, only items from whitelist are used."
    )


def parse_blacklist_args(args):
    blacklist = None
    whitelist = None
    if args.blacklist:
        state = load_marked(args.blacklist)
        blacklist = set(state.marked)
        LOG.info(f"Loaded blacklist with {len(blacklist)} items.")
    if args.whitelist:
        state = load_marked(args.whitelist)
        whitelist = set(state.marked)
        LOG.info(f"Loaded whitelist with {len(whitelist)} items.")

    return blacklist, whitelist


def add_dataset_args(parser: argparse.ArgumentParser):
    ground_truth_options = parser.add_mutually_exclusive_group(
        required=True,
    )
    ground_truth_options.add_argument(
        "-text",
        help="DEPRECATED. ASCII file with lines or sentences description."
             "Either IAM ascii format, or our JSON format.",
    )
    ground_truth_options.add_argument(
        "-gt-iam-ascii",
        help="ASCII file with lines, sentences or words ground truth in IAM format."
    )
    ground_truth_options.add_argument(
        "-gt-iam-xml",
        help="Root directory with IAM forms XML files, with ground truth description."
    )
    ground_truth_options.add_argument(
        "-gt-page-xml",
        help="PAGE-XML file with ground truth description."
    )
    ground_truth_options.add_argument(
        "-gt-json",
        help="JSON file with ground truth description."
    )

    parser.add_argument(
        "-img",
        help="Root directory of IAM image forms, lines, sentences or word files."
             " Is required if ground truth is given in one of IAM formats.",
    )
    parser.add_argument(
        "-max-ds-items",
        type=int,
        help="Maximum amount of loaded datasource items"
    )

    add_blacklist_args(parser)

    parser.add_argument(
        "-no-ignore-list",
        action="store_true",
        help="Apply ignore list from vocabulary."
    )


def parse_dataset_args(args, vocabulary: Optional[Vocabulary]=None):
    blacklist, whitelist = parse_blacklist_args(args)

    def _parse_gt_format(args) -> Tuple[Optional[GTFormat], Optional[str]]:
        if args.gt_iam_ascii:
            return GTFormat.IAM_ASCII, args.gt_iam_ascii
        if args.gt_iam_xml:
            return GTFormat.IAM_XML, args.gt_iam_xml
        if args.gt_page_xml:
            return GTFormat.PAGE_XML, args.gt_page_xml
        if args.gt_json:
            return GTFormat.JSON, args.gt_json
        if args.text:
            if Path(args.text).suffix == ".json":
                return GTFormat.JSON, args.text
            return GTFormat.IAM_ASCII, args.text
        return None, None

    gt_format, gt_path = _parse_gt_format(args)

    dataset, voc = load_dataset(
        gt_format=gt_format,
        gt_path=gt_path, img_dir=args.img,
        vocabulary=vocabulary,
        apply_ignore_list=not args.no_ignore_list,
        blacklist=blacklist,
        whitelist=whitelist,
        max_ds_items=args.max_ds_items,
    )

    save_whitelist(dataset)
    return dataset, voc
