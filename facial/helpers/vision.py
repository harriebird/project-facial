import os
import cv2 as cv
from . import config


def load_cascade(cascade):
    return cv.CascadeClassifier(os.path.join(cv.__path__[0], 'data', cascade))


def init_hog():
    return cv.HOGDescriptor(config.win_size, config.block_size, config.block_stride, config.cell_size, config.nbins)


