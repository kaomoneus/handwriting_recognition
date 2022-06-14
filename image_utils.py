import logging
from typing import Tuple, Optional

import numpy
import numpy as np
import tensorflow as tf

LOG = logging.getLogger(__name__)

"""
Defines internal format of rendered strings.
Note, if string is quite short for such aspect ration
it is supposed to pad such rendered string image.
"""
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 32

PAD_COLOR = 255


def distortion_free_resize(image: numpy.ndarray):
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


def load_and_pad_image(image_path: str, roi: Tuple[int, int, int, int] = None) -> Optional[np.ndarray]:
    image = tf.io.read_file(image_path)

    try:
        image = tf.image.decode_png(image, 1)
    except Exception as e:
        LOG.warning(f"Unable to load '{image_path}': {e}")
        return None

    if roi:
        left, top, width, height = roi
        image = tf.image.crop_to_bounding_box(image, top, left, height, width)

    image = distortion_free_resize(image)
    image = tf.cast(image, tf.float32) / 255.0
    return image
