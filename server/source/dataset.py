from .constants import IMSIZE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from keras.utils import np_utils


def build_dataset(trainset_path, labels, ncount):
    train_images = []
    train_labels = []

    for label in labels):
        label_folder_path = os.path.join(trainset_path, label)

        for image in os.listdir(label_folder_path):
            # get im path
            image_path = os.path.join(label_folder, image)

            # read & resize
            im = cv2.imread(image_path)
            im = cv2.resize(img, (IMSIZE, IMSIZE))

            # append
            train_images.append(im)
            train_labels.append(labels.index(label))

    # convert to np array
    train_images = np.array(train_images, dtype=np.float64)
    train_labels = np.array(train_labels)

    return train_labels, train_images

def split_train_val(images, labels, val_percentage=0.1):

    # Random Shuffle images and labels (in unison)
    s = np.random.permutation(len(labels))
    np.random.shuffle(s)
    images = images[s]
    labels = labels[s]

    perc_count = int(len(labels) * val_percentage)

    X_train, y_train = images[perc_count:], labels[perc_count:]
    X_val, y_val = images[:perc_count], labels[:perc_count]

    return X_train, y_train, X_val, y_val


def normalize(X_train, y_train, X_val, y_val):
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_train = X_train / 255.0
    X_val = X_val / 255.0

    y_train = np_utils.to_categorical(y_train)
    y_val = np_utils.to_categorical(y_val)

    return X_train, y_train, X_val, y_val
