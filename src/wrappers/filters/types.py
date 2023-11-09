type RGBPixel = tuple[int, int, int]
type GrayscalePixel = int

type RGBImage = list[list[RGBPixel]]
type GrayscaleImage = list[list[GrayscalePixel]]

type Image = RGBImage | GrayscaleImage