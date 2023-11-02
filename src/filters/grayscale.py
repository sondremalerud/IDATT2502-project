import numpy as np

RGBPixel = tuple[int, int, int]
GrayscalePixel = int

def vxycc709(img: list[list[RGBPixel]]) -> list[GrayscalePixel]:
    """
    Converts an image from RGB to grayscale using color weights from ITU's report on colorimetry (2018)
    Weights chosen are for the vxYCC709 color space
    https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2380-2-2018-PDF-E.pdf
    """
    RED_WEIGHT = 0.2126
    GREEN_WEIGHT = 0.7152
    BLUE_WEIGHT = 0.0722

    red_pixels = RED_WEIGHT * img[:, :, 0]
    green_pixels = GREEN_WEIGHT * img[:, :, 1]
    blue_pixels = BLUE_WEIGHT * img[:, :, 2]

    return red_pixels + green_pixels + blue_pixels

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    
    img = np.random.randint(0, 255, (8, 8, 3))
    fig = plt.figure()

    fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    
    fig.add_subplot(1, 2, 2)
    plt.imshow(rgb2grayscale(img), cmap='Greys_r')
    
    plt.show()