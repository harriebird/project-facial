import os

# Directory Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# HOG Parameters
win_size = (50, 50)
block_size = (10, 10)
block_stride = (5, 5)
cell_size = (5, 5)
nbins = 9

# Training Config
