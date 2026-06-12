import numpy as np
from PIL import Image


def resize_img(image, shape: tuple[int, int] = (64, 64)):
    return np.asarray(Image.fromarray(image).resize(shape, Image.Resampling.BILINEAR))
