import dataclasses
import logging
import os
from multiprocessing import Pool, cpu_count
from os import listdir
from typing import List, Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.python.data import AUTOTUNE
from tqdm import tqdm

from image_utils import load_and_pad_image
from text_utils import Vocabulary

"""
Training batch size
"""
BATCH_SIZE = 64

IMG_NAME_IDX = 0
VALUE_IDX = 8

LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class GroundTruthPathsItem:
    str_value: str
    img_path: str


@dataclasses.dataclass
class GroundTruthItem:
    img: np.ndarray
    str_value: str = ""
    img_path: str = None


Dataset = List[GroundTruthItem]


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


def _load_sample(gtp: GroundTruthPathsItem):
    return GroundTruthItem(
        str_value=gtp.str_value,
        img=load_and_pad_image(gtp.img_path),
        img_path=gtp.img_path
    )


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
    res = []
    vocabulary = Vocabulary()

    locs = get_img_locs(img_dir)

    with open(str_values_file_path, "r") as f:
        with vocabulary.loader() as characters:

            lines = [ll for ll in f.readlines() if not ll.startswith("#")]
            LOG.debug(f"Total lines/sentences read: {len(lines)}")

            load_tasks = []

            for line in lines:
                l_items = line.split(" ")
                str_value = l_items[VALUE_IDX].strip().replace("|", " ")

                if len(str_value) > max_word_len:
                    LOG.warning(f"Skipping word '{str_value}', exceeds max length {max_word_len} characters.")
                    continue

                img_name = l_items[IMG_NAME_IDX]
                img_path = locs[img_name]

                characters.update(str_value)

                # characters.update(str_value)
                load_tasks.append(GroundTruthPathsItem(
                    str_value=str_value,
                    img_path=img_path
                ))

            pool = Pool(processes=(cpu_count() * 4))

            for gt in tqdm(
                pool.imap_unordered(_load_sample, load_tasks),
                total=len(load_tasks),
                desc="Loading rendered text"
            ):
                if gt.img is None:
                    LOG.warning(f"Skipping: '{gt.img_path}")
                    continue
                res.append(gt)

    return res, vocabulary


def extract_images_and_labels(ds: Dataset) -> Tuple[List[np.ndarray], List[str]]:
    images = []
    labels = []

    for gt in ds:
        labels.append(gt.str_value)
        images.append(gt.img)

    return images, labels


def tf_dataset(ds: Dataset, vocabulary: Vocabulary) -> tf.data.Dataset:
    """
    Converts dataset to internal tensorflow representation
    :param ds:
    :param image_size: size of image in format <width, height>
    :return: tf.Data.Dataset instance
    """

    images, labels = extract_images_and_labels(ds)

    def _process_images_labels(image, label):
        label = vocabulary.vectorize_label(label)
        return {"image": tf.convert_to_tensor(image, dtype=tf.float32), "label": label}

    tf_ds = tf.data.Dataset.from_tensor_slices(
        (images, labels)
    ).map(_process_images_labels)

    return tf_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE).cache()
