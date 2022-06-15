import dataclasses
import logging
import os
from os import listdir
from typing import List, Dict, Tuple

import tensorflow as tf
from tensorflow.python.data import AUTOTUNE

from image_utils import tf_distortion_free_resize
from text_utils import Vocabulary

"""
Training batch size
"""
BATCH_SIZE = 64

IMG_NAME_IDX = 0
VALUE_IDX = 8

LOG = logging.getLogger(__name__)

ROI = Tuple[int, int, int, int]


@dataclasses.dataclass
class GroundTruthPathsItem:
    str_value: str
    img_path: str
    roi: ROI = None


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

    # TODO: preprocess image and same them in tmp directory.
    #   This is IMPORTANT. Currently network shows perfect results
    #   on IAM, but very poor results otherwise.
    #   We can't preprocess them in tf_distortion_free_resize
    #   just because tf function is very special
    #   it uses only TF calls it thus might be converted into
    #   internal TF graphs.
    # with TemporaryDirectory() as tmp:
    #     # Run required preprocessing:
    #     #    different sorts of degradation etc.
    #     pass

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
                    img_path=img_path
                ))

    return res, vocabulary


def extract_images_and_labels(ds: Dataset) -> Tuple[List[str], List[str], List[ROI]]:
    paths = []
    labels = []
    rois = []

    for gt in ds:
        labels.append(gt.str_value)
        paths.append(gt.img_path)
        rois.append(gt.roi)

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
        image = tf.image.decode_png(img_bytes, 1)
        if roi is not None:
            image = tf.image.crop_to_bounding_box(image, roi[1], roi[0], roi[3], roi[2])

        image = tf_distortion_free_resize(image)
        image = tf.cast(image, tf.float32) / 255.0

        return {"image": image, "label": label}

    tf_ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels, rois)
    ).map(_process_images_labels, num_parallel_calls=AUTOTUNE)

    return tf_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE).cache().shuffle(len(ds))
