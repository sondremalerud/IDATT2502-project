import numpy as np

def divide(img: np.ndarray, factor: int):
    """Divides the resolution of the provided image by an integer factor"""
    assert len(img.shape) == 2 or img.shape[2] == 1, "This filter should only be used on grayscale images, provided image appears to be RGB"

    assert isinstance(factor, int), "Can only rescale by integer factor"

    M: int
    N: int

    if len(img.shape) == 2:
        M, N = img.shape
    else:
        M, N, _ = img.shape

    MK = M // factor
    NL = N // factor

    return img[:MK*factor, :NL*factor].reshape(MK, factor, NL, factor, 1).mean(axis=(1, 3)).astype(np.uint8)
