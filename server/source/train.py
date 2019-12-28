#!/usr/bin/env python
# coding: utf-8

# LOCALS
from .constants import IMSIZE, DATASET_PATH, trainset_path, testset_path
import .dataviz as dv
import .dataset as ds
import .stats as st
from .model import create_model

import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys


def download_ds(ds_path):
    if not (os.path.isdir(DATASET_PATH)):
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("alexattia/the-simpsons-characters-dataset")

        get_ipython().getoutput("""
unzip the-simpsons-characters-dataset.zip -d the-simpsons-characters-dataset
        """)


def train(dataset_path, output_file, epochs):

    # setup paths
    download_ds(dataset_path)
    TRAINSET_PATH = trainset_path(dataset_path)
    TESTSET_PATH = trainset_path(dataset_path)


    LABELS = os.listdir(TRAINSET_PATH)
    LABELS.remove('simpsons_dataset')

    # get stats from dataset
    mean, sd, q1, q3 = st.get_stats(LABELS)
    iqr = q3 - q1

    low_threshold = q1
    upper_threshold = q3

    # clean dataset
    count, labels = st.clean_dataset(low_threshold, upper_threshold, LABELS)


    # build dataset
    _labels, _images = ds.build_dataset(labels, count)
    X_train, y_train, X_val, y_val = ds.split_train_val(_images, _labels)
    X_train_n, y_train_n, X_val_n, y_val_n = ds.normalize(X_train, y_train, X_val, y_val)


    # create & compile & fit & save the model
    model = create_model(input_shape=X_train_n.shape[1:],
                        class_num=len(labels))
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    model.fit(X_train_n, y_train_n,
            validation_data=(X_val_n, y_val_n),
            epochs=epochs, batch_size=64)

    model.save(output_file)

    # model evaluation
    scores = model.evaluate(X_val_n, y_val_n, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
Train simpsons image recognition model with tensorflow.
""")

    parser.add_argument('--dataset_path', '-d', metavar='D', required=True,
                        help='Dataset path', dest='dataset_path')
    parser.add_argument('--output_file', '-o', metavar='D', required=True,
                        help='Output file', dest='output_file')
    parser.add_argument('--epochs', '-e', metavar='E', default=30,
                        help='Epochs to be trained', dest='epochs')
    args = parser.parse_args()

    train(args.dataset_path, args.output_file, args.epochs)
