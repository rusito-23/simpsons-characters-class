#!/usr/bin/env python
# coding: utf-8

import os
import glob
import numpy as np
from data_utils import load_utils


def load_test_set(testset_path, labels):
    pics = glob.glob(os.path.join(testset_path, '*.*'))
    pics = sorted(pics)
    test_labels = ['_'.join(p.split('/')[-1].split('_')[:-1]) for p in pics]

    # load test images and the categorical labels
    X_test = load_utils.X_load(pics)
    y_test = load_utils.y_load(test_labels, labels)

    # as the number of test labels is not the same as the model labels
    # (we have less labels than the dataset)
    t_labels = np.unique(np.array([labels.index(i) for i in test_labels]))
    t_names = np.unique(np.array(test_labels))

    return X_test, y_test, t_labels, t_names
