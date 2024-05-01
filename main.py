import keras
import tensorflow as tf
import numpy as np
from images import *

class_names = ['L-T_l', 'L-T_m', 'L-T_h', 'patch_l', 'patch_m', 'patch_h',
               'pothole_l', 'pothole_m', 'pothole_h', 'R-W_l', 'R-W_m', 'R-W_h', 'Rutting']

images = []
labels = []