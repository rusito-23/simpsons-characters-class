#!/usr/bin/env python
# coding: utf-8

import os

# PATHS

DATASET_PATH = 'the-simpsons-characters-dataset/'
TRAINSET_PATH = 'simpsons_dataset/'
TESTSET_PATH = 'kaggle_simpson_testset/kaggle_simpson_testset/'

def trainset_path(base):
    os.path.join(base, TRAINSET_PATH)

def testset_path(base):
    os.path.join(base, TESTSET_PATH)

# CONSTANTS

IMSIZE = 50
