from .constants import TRAINSET_PATH, IMSIZE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2


def build_dataset(labels, ncount):
    train_images = []
    train_labels = []

    for i in range(len(labels)):

        label = labels[i]
        count = int(ncount[i])

        label_folder = os.path.join(TRAINSET_PATH, label)

        _images = os.listdir(label_folder)
        for j in range(count):
            _image_path = os.path.join(label_folder, _images[j])

            img = cv2.imread(_image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMSIZE, IMSIZE))

            train_images.append(img)
            train_labels.append(labels.index(label))

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
    
    
    