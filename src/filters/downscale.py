import numpy as np

def divide(img, factor: int):
    """Divides the resolution of the provided image by an integer factor"""
    assert len(img.shape) == 2, "This filter should only be used on grayscale images, provided image appears to be RGB"

    assert isinstance(factor, int), "Can only rescale by integer factor"

    M, N = img.shape

    MK = M // factor
    NL = N // factor

    return img[:MK*factor, :NL*factor].reshape(MK, factor, NL, factor).mean(axis=(1, 3)).astype(np.uint8)
