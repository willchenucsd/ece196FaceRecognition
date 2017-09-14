"""
ECE196 Face Recognition Project
Author: W Chen

Use this as a template to:
1. load weights for vgg16
2. load images
3. finetune network
4. save weights
"""


from keras.models import Model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
import numpy as np
import glob
import os
import cv2
import random


def load_model():
    
    return model


def main():
    load_model()
    return


if __name__ == '__main__':
    main()
