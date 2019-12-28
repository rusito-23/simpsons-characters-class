#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from .constants import TRAINSET_PATH
import matplotlib.pyplot as plt


def get_stats(labels):
    # Get the image count for every label
    count = [len(os.listdir(TRAINSET_PATH + l)) for l in labels]
    count_np = np.array(count)

    # Get importants stats from this array
    mean = np.mean(count_np)
    sd = np.std(count_np)

    # get
    count_df = pd.DataFrame(count_np)
    quantiles = count_df.quantile([0.25, 0.75]).to_numpy()

    q1 = quantiles[0][0]
    q3 = quantiles[1][0]

    print(f"""
STATS:
    [Mean] -> {mean}
    [Std Deviation] -> {sd}
    [Q1] -> {q1}
    [Q3] -> {q3}
""")


    plt.plot(count)
    plt.show()

    return mean, sd, q1, q3


def clean_dataset(low_thres, upp_thres, labels):
    count = [len(os.listdir(TRAINSET_PATH + l)) for l in labels]
    _labels = []
    ncount = []

    for i in range(len(labels)):
        if count[i] < low_thres:
            print(f'removing: {labels[i]}')
            continue
        _labels.append(labels[i])

        if count[i] > upp_thres:
            ncount.append(upp_thres)
        else:
            ncount.append(count[i])

    return ncount, _labels
