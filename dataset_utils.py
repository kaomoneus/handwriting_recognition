import dataclasses
import json
import logging
import os
from multiprocessing import Pool
from os import listdir
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.data import AUTOTUNE
from tqdm import tqdm

from config import BATCH_SIZE
from image_utils import tf_distortion_free_resize, load_and_pad_image, augment_image, distortion_free_resize, \
    IMAGE_WIDTH, IMAGE_HEIGHT
from text_utils import Vocabulary

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


def load_dataset(
    str_values_file_path: str,
    img_dir: str,
    max_word_len: int
) -> Tuple[Dataset, Vocabulary]:
    """
    Loads dataset in our own format.
    We intended to use such format because it is easier to evaluate and check.
    We could use internal tensorflow format, it it might be stored out
    of python context (in native part). In this case it might be hard to check
    its contents during debug.
    :param str_values_file_path: path to ground truth values (IAM ASCII format)
    :param img_dir: root path to images directory.
    :param max_word_len: max allowed word length (restricted by network architecture)
    :return: Dataset instance (which is a list)
    """
    res: Dataset = []
    vocabulary = Vocabulary()

    locs = get_img_locs(img_dir)

    with open(str_values_file_path, "r") as f:
        with vocabulary.loader() as characters:

            lines = [ll for ll in f.readlines() if not ll.startswith("#")]
            LOG.debug(f"Total lines/sentences read: {len(lines)}")

            for line in lines:
                l_items = line.split(" ")
                str_value = l_items[VALUE_IDX].strip().replace("|", " ")

                img_name = l_items[IMG_NAME_IDX]
                img_path = locs[img_name]

                skip_msg = f"Skipping word '{str_value}', rendered as '{img_path}': %s"

                if len(str_value) > max_word_len:
                    LOG.warning(skip_msg % "exceeds max length {max_word_len} characters.")
                    continue

                if os.path.getsize(img_path) == 0:
                    LOG.warning(skip_msg % "image is empty")
                    continue

                characters.update(str_value)

                # characters.update(str_value)
                res.append(GroundTruthPathsItem(
                    str_value=str_value,
                    img_path=img_path,
                    img_name=img_name
                ))

    return res, vocabulary


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

    with open(ground_truth_file, "w") as f:
        print(src_item.str_value, file=f)

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

    if len(ds) > 100:
        LOG.info("Launching DS preprocessing pool...")
        pool = Pool(processes=(os.cpu_count() * 4))

        for d, gts in tqdm(
            pool.imap_unordered(_preprocess_item_task, preprocess_args),
            total=len(ds),
            desc="Preprocessing dataset"
        ):
            apply_preprocess_result(d, gts)
    else:
        for args in preprocess_args:
            d, gts = _preprocess_item(**args)
            apply_preprocess_result(d, gts)

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


def tf_dataset(ds: Dataset, vocabulary: Vocabulary) -> tf.data.Dataset:
    """
    Converts dataset to internal tensorflow representation
    :param ds:
    :param vocabulary: is used to vectorize labels properly
    :return: tf.Data.Dataset instance
    """

    paths, labels, rois = extract_images_and_labels(ds)

    def _process_images_labels(img_path, label, roi):
        label = vocabulary.vectorize_label(label)

        img_bytes = tf.io.read_file(img_path)
        image = tf.image.decode_image(img_bytes, 1, expand_animations=False)

        if roi[2] != 0:
            image = tf.image.crop_to_bounding_box(image, roi[1], roi[0], roi[3], roi[2])

        if image.shape[0] != IMAGE_WIDTH and image.shape[1] != IMAGE_HEIGHT:
            image = tf_distortion_free_resize(image)

        image = tf.cast(image, tf.float32) / 255.0

        return {"image": image, "label": label}

    tf_ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels, rois)
    ).map(_process_images_labels, num_parallel_calls=AUTOTUNE)

    return tf_ds.shuffle(len(ds)).batch(BATCH_SIZE).prefetch(AUTOTUNE).cache()
