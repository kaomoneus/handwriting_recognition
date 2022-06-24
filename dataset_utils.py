import dataclasses
import json
import logging
import os
import pathlib
from multiprocessing import Pool
from os import listdir
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Iterable, Union, Callable

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.data import AUTOTUNE
from tqdm import tqdm

from config import BATCH_SIZE_DEFAULT, MAX_WORD_LEN_DEFAULT
from errors import Error
from image_utils import tf_distortion_free_resize, load_and_pad_image, augment_image, distortion_free_resize, \
    IMAGE_WIDTH, IMAGE_HEIGHT
from text_utils import Vocabulary, add_voc_args

GROUND_TRUTH_FILENAME = "ground_truth.txt"

IMG_NAME_IDX = 0
VALUE_IDX = 8

LOG = logging.getLogger(__name__)

ROI = Tuple[int, int, int, int]


@dataclasses.dataclass
class GroundTruthPathsItem:
    """
    Ground truth value
    """
    str_value: str

    """
    Unique image name
    """
    img_name: str

    """
    Full path to rendered value
    """
    img_path: str

    """
    Region of interest
    """
    roi: Optional[ROI] = None


Dataset = List[GroundTruthPathsItem]
GroundTruth = Dict[str, GroundTruthPathsItem]


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


def save_ground_truth_json(ground_truth: Dict[str, GroundTruthPathsItem], path: Union[str, os.PathLike]):
    with open(path, "w") as f:
        d = {
            img_path: dataclasses.asdict(gt)
            for img_path, gt in ground_truth.items()
        }
        json.dump(d, f, indent=4)


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


def load_dataset(
    str_values_file_path: str,
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
    :param str_values_file_path: path to ground truth values (IAM ASCII format)
    :param img_dir: root path to images directory.
    :param vocabulary: vocabulary if provided, then it will be used to filter
       words which are too long, or which use inappropriate characters
    :param apply_ignore_list: apply ignore list from vocabulary
    :param blacklist: set of blacklisted dataset items
    :param max_ds_items: dataset size limit
    :return: Dataset instance (which is a list)
    """
    max_word_len = vocabulary.max_len if vocabulary else MAX_WORD_LEN_DEFAULT
    auto_voc = Vocabulary()

    ignore_list = set(vocabulary.ignore) if apply_ignore_list else None

    allowed_characters = set(vocabulary.characters) if vocabulary else None

    num_skipped = 0

    with auto_voc.builder() as characters:
        def on_sample(gt: GroundTruthPathsItem):
            nonlocal num_skipped
            skip_msg = f"Skipping word '{gt.str_value}', rendered as '{gt.img_path}': %s"

            if whitelist and gt.img_name not in whitelist:
                # Skipped word means that we skip something from whitelist.
                # So, don't report this word is skipped.
                # num_skipped += 1
                return False

            if len(gt.str_value) > max_word_len:
                LOG.debug(skip_msg % f"exceeds max length {max_word_len} characters.")
                num_skipped += 1
                return False

            if ignore_list and gt.str_value in ignore_list:
                LOG.debug(skip_msg % "is in ignore list")
                num_skipped += 1
                return False

            if blacklist and gt.img_name in blacklist:
                LOG.debug(skip_msg % "is in blacklist")
                num_skipped += 1
                return False

            if allowed_characters:
                disallowed = set(gt.str_value).difference(allowed_characters)
                if disallowed:
                    LOG.debug(skip_msg % f"contains disallowed characters: {''.join(disallowed)}")
                    num_skipped += 1
                    return False

            if os.path.getsize(gt.img_path) == 0:
                LOG.debug(skip_msg % "image is empty")
                num_skipped += 1
                return False

            characters.update(gt.str_value)
            return True

        if pathlib.Path(str_values_file_path).suffix == ".json":
            ground_truth = load_ground_truth_json(str_values_file_path, on_sample, max_ds_items)
            res = ground_truth_to_dataset(ground_truth), auto_voc
        elif img_dir:
            res = load_iam_dataset(str_values_file_path, img_dir, on_sample, max_ds_items), auto_voc
        else:
            raise Error(f"Text file is in IAM format, but images directory is not provided")

    LOG.info(f"Total amount of skipped words: {num_skipped}")
    return res


def _preprocess_item(
    src_item: GroundTruthPathsItem,
    cache_dir: str,
    only_threshold: bool = False,
    keep_existing_augmentations: bool = False,
    ignore_augmentations: Set[str] = None
):
    img_path = src_item.img_path
    roi = src_item.roi

    cache_subdir = Path(cache_dir) / src_item.img_name
    ext = Path(img_path).suffix

    ground_truth_file = cache_subdir / GROUND_TRUTH_FILENAME

    if keep_existing_augmentations and cache_subdir.exists():
        if ground_truth_file.exists():
            with open(ground_truth_file, "r") as f:
                str_value = f.readline().strip()
        else:
            str_value = ""

        res = [
            GroundTruthPathsItem(
                str_value=str_value,
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

    if roi:
        left, top, width, height = roi
        image = image[top:top+height, left:left+width, :]

    cache_subdir.mkdir(parents=True, exist_ok=True)

    # Apply different sorts of threshold
    augmentation = augment_image(image, only_threshold)

    if not only_threshold and ignore_augmentations:
        for aug in ignore_augmentations:
            augmentation.pop(aug)

    res: Dataset = []

    for aug_name, aug in augmentation.items():
        aug_name = f"{src_item.img_name}-{aug_name}"
        dest_path = cache_subdir / f"{aug_name}{ext}"
        cv2.imwrite(str(dest_path), distortion_free_resize(aug))
        res.append(GroundTruthPathsItem(
            str_value=src_item.str_value,
            img_name=aug_name,
            img_path=str(dest_path),
            roi=None
        ))

    return src_item, res


def _preprocess_item_task(args_dict):
    return _preprocess_item(**args_dict)


def preprocess_dataset(
    ds: Dataset,
    only_threshold: bool,
    cache_dir: str,
    keep: bool = False,
    full: bool = False,
) -> Dataset:
    """
    Runs image preprocessing and augmentation
    :param ds: source dataset
    :param only_threshold: don't create augmented images, only apply threshold
    :param cache_dir: cache directory where preprocessing results will be stored
    :param keep: use cached items instead of making new ones
    :param full: also include blured and adaptive thresholded (and may be more)
    :return: modified dataset with paths targeting to cache directory
    """

    res: Dataset = []

    preprocess_args = [dict(
        src_item=d,
        cache_dir=cache_dir,
        only_threshold=only_threshold,
        keep_existing_augmentations=keep,
        ignore_augmentations={"blured", "adaptive_threshold"} if not full else None
    ) for d in ds]

    def apply_preprocess_result(d, gts):
        if full:
            res.append(d)
        res.extend(gts)

    if len(ds) > 10000:
        LOG.info("Launching DS preprocessing pool...")
        pool = Pool(processes=(os.cpu_count() * 4))

        for d, gts in tqdm(
            pool.imap_unordered(_preprocess_item_task, preprocess_args),
            total=len(ds),
            desc="Preprocessing dataset (async)"
        ):
            apply_preprocess_result(d, gts)
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


def add_dataset_args(parser):
    parser.add_argument(
        "-text",
        required=True,
        help="ASCII file with lines or sentences description. Either IAM ascii format, or our JSON format.",
    )
    parser.add_argument(
        "-img",
        help="Root directory of IAM image lines or sentences files."
             " Is required if file with lines given in IAM format.",
    )
    parser.add_argument(
        "-max-ds-items",
        type=int,
        help="Maximum amount of loaded datasource items"
    )
    parser.add_argument(
        "-blacklist",
        help="File with blacklisted item names. Should be in format of 'ploti' state."
    )
    parser.add_argument(
        "-whitelist",
        help="File with whitelist item names. Should be in format of 'ploti' state."
             " If specified, only items from whitelist are used."
    )
    parser.add_argument(
        "-no-ignore-list",
        action="store_true",
        help="Apply ignore list from vocabulary."
    )


def parse_dataset_args(args, vocabulary: Vocabulary):
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

    return load_dataset(
        str_values_file_path=args.text, img_dir=args.img,
        vocabulary=vocabulary,
        apply_ignore_list=not args.no_ignore_list,
        blacklist=blacklist,
        whitelist=whitelist,
        max_ds_items=args.max_ds_items,
    )
