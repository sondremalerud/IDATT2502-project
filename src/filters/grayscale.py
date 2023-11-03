import numpy as np
from typing import Literal

RGBPixel = tuple[int, int, int]
GrayscalePixel = int

def vxycc709(img: list[list[RGBPixel]]) -> list[GrayscalePixel]:
    """Converts an image from RGB to grayscale using color weights from ITU's report on colorimetry (2018)
    Weights chosen are for the vxYCC709 color space
    https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2380-2-2018-PDF-E.pdf"""
    RED_WEIGHT = 0.2126
    GREEN_WEIGHT = 0.7152
    BLUE_WEIGHT = 0.0722

    red_pixels = RED_WEIGHT * img[:, :, 0]
    green_pixels = GREEN_WEIGHT * img[:, :, 1]
    blue_pixels = BLUE_WEIGHT * img[:, :, 2]

    return (red_pixels + green_pixels + blue_pixels).astype(np.uint8)


def vxycc601(colored_image):
    """Converts an image from RGB to grayscale using color weights from ITU's report on colorimetry (2018)
    Weights chosen are for the vxYCC601 color space
    https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2380-2-2018-PDF-E.pdf"""
    weighted_red = 0.299 * colored_image[:, :, 0]
    weighted_blue = 0.587 * colored_image[:, :, 1]
    weighted_green = 0.114 * colored_image[:, :, 2]
    desaturated_image = weighted_red + weighted_green + weighted_blue
    return desaturated_image.astype(np.uint8)


def average(colored_image):
    """Converts an image from RGB to grayscale by averaging the color channels"""
    red = colored_image[:, :, 0]
    blue = colored_image[:, :, 1]
    green = colored_image[:, :, 2]
    desaturated_image = red + green + blue
    desaturated_image = desaturated_image/3
    return desaturated_image.astype(np.uint8)


def _channel(img: list[list[RGBPixel]], channel: Literal[0, 1, 2]) -> list[GrayscalePixel]:
    """Returns a specific color channel from the image"""
    return img[:, :, channel]


def red_channel(img: list[list[RGBPixel]]) -> list[GrayscalePixel]:
    """Returns the red channel of the image"""
    return _channel(img, 0)


def green_channel(img: list[list[RGBPixel]]) -> list[GrayscalePixel]:
    """Returns the green channel of the image"""
    return _channel(img, 1)


def blue_channel(img: list[list[RGBPixel]]) -> list[GrayscalePixel]:
    """Returns the blue channel of the image"""
    return _channel(img, 2)

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    
    img = np.random.randint(0, 255, (8, 8, 3))
    fig = plt.figure()

    fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    
    fig.add_subplot(1, 2, 2)
    plt.imshow(vxycc709(img), cmap='Greys_r')
    
    plt.show()