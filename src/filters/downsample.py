import numpy as np

def _downsample(pixel, depth):
    return pixel // (256 // depth)

def _upsample(pixel, factor):
    return pixel * factor

downsample = np.vectorize(_downsample)
upsample = np.vectorize(_upsample)