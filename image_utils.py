import logging
from typing import Tuple, Optional, Dict

import numpy
import numpy as np
import tensorflow as tf
import cv2

LOG = logging.getLogger(__name__)

"""
Defines internal format of rendered strings.
Note, if string is quite short for such aspect ration
it is supposed to pad such rendered string image.
"""
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 32

PAD_COLOR = 255


def _image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Copyright to https://stackoverflow.com/a/44659589/5160481
    """
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        rw = width / float(w)
        rh = height / float(h)

        dim = (width, int(h * rw)) if rw < rh else (int(w * rh), height)

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def distortion_free_resize(image: numpy.ndarray):
    """
    ### Resizing images without distortion

    Instead of square images, many OCR models work with rectangular images. This will become
    clearer in a moment when we will visualize a few samples from the dataset. While
    aspect-unaware resizing square images does not introduce a significant amount of
    distortion this is not the case for rectangular images. But resizing images to a uniform
    size is a requirement for mini-batching. So we need to perform our resizing such that
    the following criteria are met:

    * Aspect ratio is preserved.
    * Content of the images is not affected.
    :param image source image to be resized
    :return resized image
    """
    w, h = IMAGE_WIDTH, IMAGE_HEIGHT

    # TODO: after resize in order to prevent corner-looking artifacts
    #    perform some bluring and thresholding
    image = _image_resize(image, w, h, cv2.INTER_CUBIC)
    if len(image.shape) == 2:
        image = image.reshape([*image.shape, 1])

    # Check tha amount of padding needed to be done.
    pad_height = h - image.shape[0]
    pad_width = w - image.shape[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = np.pad(
        image,
        pad_width=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
        constant_values=PAD_COLOR
    )

    # image = tf.transpose(image, perm=[1, 0, 2])
    # image = tf.image.flip_left_right(image)
    return image


def tf_distortion_free_resize(image):
    """
    Tensorflow (as given in original article) implementation of `distortion_free_resize`.
    :param image source image to be resized
    :return: resized image
    """
    w, h = IMAGE_WIDTH, IMAGE_HEIGHT
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
        constant_values=PAD_COLOR
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image


def load_and_pad_image(
    image_path: str,
    roi: Tuple[int, int, int, int] = None,
) -> Optional[np.ndarray]:
    try:
        image: np.ndarray = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
        if image is None:
            LOG.warning(f"Unable to load '{image_path}'")
            return None
    except Exception as e:
        LOG.warning(f"Unable to load '{image_path}': {e}")
        return None

    if len(image.shape) == 2:
        image = image.reshape([*image.shape, 1])

    if roi:
        left, top, width, height = roi
        image = image[top:top+height, left:left+width, :]

    image = distortion_free_resize(image)

    return image


def augment_image(src_img: np.ndarray, only_threshold: bool) -> Dict[str, np.ndarray]:
    """
    Augments image
    :param src_img: image to be augmented
    :param only_threshold: only apply adaptive threshold
    :return: dictionary where key is augmentation name, value is augmented image
    """
    res = {}

    blured = cv2.GaussianBlur(src_img, (3, 3), sigmaX=1.)
    threshold_value, thr_mid = cv2.threshold(blured, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    res["threshold"] = thr_mid
    if only_threshold:
        return res

    thr_val_left = int(threshold_value * 0.95)
    thr_val_right = int(threshold_value * 1.2)

    _, thr_left = cv2.threshold(blured, thr_val_left, 255, cv2.THRESH_BINARY)
    _, thr_right = cv2.threshold(blured, thr_val_right, 255, cv2.THRESH_BINARY)
    adaptive_threshold = cv2.adaptiveThreshold(
        src_img,
        255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3
    )

    res.update({
        "blured": blured,
        "adaptive_threshold": adaptive_threshold,
        "threshold_left": thr_left,
        "threshold_right": thr_right,
    })
    return res
