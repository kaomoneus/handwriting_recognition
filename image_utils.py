import logging
from typing import Tuple, Optional

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


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
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
    w, h = IMAGE_WIDTH, IMAGE_HEIGHT

    # TODO: after resize in order to prevent corner-looking artifacts
    #    perform some bluring and thresholding
    image = image_resize(image, w, h, cv2.INTER_CUBIC)
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


def load_and_pad_image(
    image_path: str,
    roi: Tuple[int, int, int, int] = None,
    applyThreshold = False
) -> Optional[np.ndarray]:
    try:
        image: np.ndarray = cv2.imread(image_path)
        if image is None:
            LOG.warning(f"Unable to load '{image_path}'")
            return None
    except Exception as e:
        LOG.warning(f"Unable to load '{image_path}': {e}")
        return None

    if len(image.shape) == 2:
        image = image.reshape([*image.shape, 1])
    elif image.shape[2] != 1:
        image: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.reshape([*image.shape, 1])

    if roi:
        left, top, width, height = roi
        image = image[top:top+height, left:left+width, :]

    if applyThreshold:
        # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image = image.reshape([*image.shape, 1])

    image = distortion_free_resize(image)

    image = tf.cast(image, tf.float32) / 255.0
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)

    return image
