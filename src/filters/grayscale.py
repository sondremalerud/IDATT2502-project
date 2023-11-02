import numpy as np
from filtertypes import RGBPixel, GrayscalePixel

def rgb2grayscale(img: list[list[RGBPixel]]) -> list[GrayscalePixel]:
    """
        Converts an image from RGB to grayscale using color weights from ITU's report on colorimetry (2018)
        https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2380-2-2018-PDF-E.pdf
    """
    RED_WEIGHT = 0.2126
    GREEN_WEIGHT = 0.7152
    BLUE_WEIGHT = 0.0722

    WEIGHTS = tuple([RED_WEIGHT, GREEN_WEIGHT, BLUE_WEIGHT])
   
    output = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    for row, pixels in enumerate(img):
        for col, pixel in enumerate(pixels):
            output[row][col] = int(sum([channel * weight for (channel, weight) in zip(pixel, WEIGHTS)]))

    return output

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    
    img = np.random.randint(0, 255, (8, 8, 3))
    fig = plt.figure()

    fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    
    fig.add_subplot(1, 2, 2)
    plt.imshow(rgb2grayscale(img), cmap='Greys_r')
    
    plt.show()