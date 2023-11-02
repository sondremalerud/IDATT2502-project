def eighth_gray(img):
    assert len(img.shape) == 2, "This filter should only be used on grayscale images, provided image appears to be RGB"

    M, N = img.shape
    STRIDE = 8

    MK = M // STRIDE
    NL = N // STRIDE

    return img[:MK*STRIDE, :NL*STRIDE].reshape(MK, STRIDE, NL, STRIDE).mean(axis=(1, 3))
