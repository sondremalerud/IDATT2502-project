RGBPixel = tuple[int, int, int]
GrayscalePixel = int

RGBImage = list[list[RGBPixel]]
GrayscaleImage = list[list[GrayscalePixel]]

Image = RGBImage | GrayscaleImage