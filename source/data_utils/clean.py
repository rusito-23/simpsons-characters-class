#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
from config import TRAINSET_PATH


def clean_labels(dataset_path, labels):
    # find quartiles
    count_per_label = [len(os.listdir(p))
                       for p in [os.path.join(dataset_path, TRAINSET_PATH, lb)
                       for lb in labels]]
    df_count = pd.DataFrame(count_per_label)
    stats = df_count.describe()
    q1, q3 = stats[0]['25%'], stats[0]['75%']
    lower_bound, upper_bound = q1, q3
    # clean labels
    ncount, nlabels = [], []
    for c, l in zip(count_per_label, labels):
        if c < lower_bound:
            continue
        nlabels.append(l)
        ncount.append(upper_bound if c > upper_bound else c)

    return ncount, nlabels
