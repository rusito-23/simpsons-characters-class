#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from config import IMSIZE


def load_pic(pic_path):
    pic = cv2.imread(pic_path)
    pic = cv2.resize(pic, (IMSIZE, IMSIZE))
    return pic / 255


def X_load(x):
    x = np.array([load_pic(impath) for impath in x])
    return x


def y_load(y, labels):
    y = np.array([labels.index(l) for l in y])
    y = to_categorical(y, len(labels))
    return y
