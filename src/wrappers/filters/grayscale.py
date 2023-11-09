import numpy as np
from typing import Literal

from wrappers.filters.types import GrayscaleImage, RGBImage


def vxycc709(img: RGBImage) -> GrayscaleImage:
    """Converts an image from RGB to grayscale using color weights from ITU's report on colorimetry (2018)
    Weights chosen are for the vxYCC709 color space
    https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2380-2-2018-PDF-E.pdf"""
    RED_WEIGHT = 0.2126
    GREEN_WEIGHT = 0.7152
    BLUE_WEIGHT = 0.0722

    red_pixels = RED_WEIGHT * img[:, :, 0]
    green_pixels = GREEN_WEIGHT * img[:, :, 1]
    blue_pixels = BLUE_WEIGHT * img[:, :, 2]

    res = red_pixels + green_pixels + blue_pixels
    res = res.reshape((img.shape[0], img.shape[1], 1))
    res = res.astype(np.uint8)
    return res


def vxycc601(img: RGBImage) -> GrayscaleImage:
    """Converts an image from RGB to grayscale using color weights from ITU's report on colorimetry (2018)
    Weights chosen are for the vxYCC601 color space
    https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2380-2-2018-PDF-E.pdf"""

    weighted_red = 0.299 * img[:, :, 0]
    weighted_blue = 0.587 * img[:, :, 1]
    weighted_green = 0.114 * img[:, :, 2]

    desaturated_image = weighted_red + weighted_green + weighted_blue

    res = desaturated_image.reshape((img.shape[0], img.shape[1], 1))
    res = res.astype(np.uint8)
    return res


def average(img: RGBImage) -> GrayscaleImage:
    """Converts an image from RGB to grayscale by averaging the color channels"""

    red = img[:, :, 0]
    blue = img[:, :, 1]
    green = img[:, :, 2]

    desaturated_image = red + green + blue
    desaturated_image = desaturated_image / 3

    res = desaturated_image.reshape((img.shape[0], img.shape[1], 1))
    res = res.astype(np.uint8)
    return res


def _channel(
    img: RGBImage, channel: Literal[0, 1, 2]
) -> GrayscaleImage:
    """Returns a specific color channel from the image"""
    return img[:, :, channel].reshape((img.shape[0], img.shape[1], 1))


def red_channel(img: RGBImage) -> GrayscaleImage:
    """Returns the red channel of the image"""
    return _channel(img, 0)


def green_channel(img: RGBImage) -> GrayscaleImage:
    """Returns the green channel of the image"""
    return _channel(img, 1)


def blue_channel(img: RGBImage) -> GrayscaleImage:
    """Returns the blue channel of the image"""
    return _channel(img, 2)
