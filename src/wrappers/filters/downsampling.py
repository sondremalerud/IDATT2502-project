import numpy as np

@np.vectorize
def downsample(pixel, depth):
    return pixel // (256 // depth)

@np.vectorize
def upsample(pixel, factor):
    return pixel * factor
