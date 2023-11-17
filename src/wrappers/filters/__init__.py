from enum import Enum

from .downsampling import downsample
from .downscale import divide
from .grayscale import (
    vxycc601,
    vxycc709,
    average,
    red_channel,
    green_channel,
    blue_channel,
)


class GrayscaleFilters(Enum):
    YCC601 = vxycc601
    YCC709 = vxycc709
    AVERAGE = average
    RED = red_channel
    GREEN = green_channel
    BLUE = blue_channel
