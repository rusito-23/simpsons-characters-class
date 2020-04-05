#!/usr/bin/env python
# coding: utf-8

import os
import glob
import itertools
import logging
from sklearn.model_selection import train_test_split as split_train_val
from data_utils.clean import clean_labels
from data_utils import load_utils
from config import TRAINSET_PATH

log = logging.getLogger('simpsons')


class SimpsonsDataset:
    """
    Kaggle Simpsons Character Dataset:
    create a Kaggle Simpsons Dataset using generators for train and val.
    Auto splits, normalizes using 255 division.
    Takes as parameter the batch size and the split val percentage.
    """

    def __init__(self, dataset_path, batch_size=1, split=0.3):
        # read all labels
        data_path = os.path.join(dataset_path, TRAINSET_PATH)
        labels = os.listdir(data_path)
        labels.remove('simpsons_dataset')

        # clean dataset
        self.label_count, self.labels = clean_labels(dataset_path, labels)
        log.debug(self.label_count, self.labels)

        # read images
        data = []
        for count, label in zip(self.label_count, self.labels):
            label_path = os.path.join(dataset_path, TRAINSET_PATH, label, '*')
            images = glob.glob(label_path)
            for i, impath in enumerate(images):
                if i > count:
                    continue
                data.append((impath, label))
        log.debug(f'Total data: {len(data)}')

        # split into train and val sets
        X = [d[0] for d in data]
        Y = [d[1] for d in data]
        X_train, X_val, y_train, y_val = split_train_val(X, Y, test_size=split)
        self.train = list(zip(X_train, y_train))
        self.val = list(zip(X_val, y_val))
        self.batch_size = batch_size
        self.train_steps = len(self.train) / batch_size
        self.val_steps = len(self.val) / batch_size
        self.num_classes = len(self.labels)
        log.debug(f'Train steps: {self.train_steps} '
                  f'Val steps: {self.val_steps}')

    def generator(self, data):
        cycle = itertools.cycle(data)
        while True:
            X, Y = [], []
            for _ in range(self.batch_size):
                impath, label = next(cycle)
                Y.append(label)
                X.append(impath)
            X = load_utils.X_load(X)
            Y = load_utils.y_load(Y, self.labels)
            yield X, Y

    def train_gen(self):
        return self.generator(self.train)

    def val_gen(self):
        return self.generator(self.val)
