#!/usr/bin/env python
# coding: utf-8

import os
import logging
import argparse
import numpy as np
from io import StringIO
import pandas as pd
from model.vanilla import classification_model as cm
from sklearn.metrics import classification_report as class_report
from data_utils.testset import load_test_set
from data_utils.dataviz import test_confusion_matrix, samples_viz
from log import config_logger
from config import IN_SHAPE
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config_logger()
log = logging.getLogger('simpsons')


def test(model_path, weights, testset_path):

    # read labels
    with open(os.path.join(model_path, 'labels.txt'), 'r') as file:
        content = file.read()
        labels = content.split('\n')
        file.close()
    log.info(f'Labels loaded succesfully - n_classes: {len(labels)}')

    # create model
    model = cm(input_shape=IN_SHAPE, class_num=len(labels))
    model.load_weights(weights)
    log.info('Model loaded succesfully')

    # load the test set
    # we need the test labels for the classification report
    # and the confusion matrix
    X_test, y_test, t_labels, t_names = load_test_set(testset_path, labels)
    y_pred = model.predict(np.array(X_test))
    log.info(f'Test on: {len(t_names)} labels')

    # get the test true labex index
    # and the predictions argmax index
    y_test_true = np.where(y_test > 0)[1]
    y_pred_argmax = np.argmax(y_pred, axis=1)

    # write results per image csv
    # for every test image, we write:
    # image_name / prediction label name / score
    results_file_path = os.path.join(model_path, 'test_results.csv')
    test_images = sorted(os.listdir(testset_path))
    results = 'Image, Prediction, Score \n'
    for i, image in enumerate(test_images):
        pred_label = labels[y_pred_argmax[i]]
        pred_score = max(y_pred[i])
        results += (f'\n {image}, {pred_label}, {pred_score:.2}\n')
    result_data = StringIO(results)
    results_df = pd.read_csv(result_data, sep=',')
    results_df.to_csv(results_file_path)
    log.info(f'Results per pic written in {results_file_path}')

    # write the classification report
    # this report includes only the test labels
    # which are less than the model labels
    report_file_path = os.path.join(model_path, 'class_report.tsv')
    report = class_report(y_test_true,
                          y_pred_argmax,
                          labels=t_labels,
                          target_names=t_names,
                          zero_division=False)
    with open(report_file_path, 'w') as report_file:
        report_file.write(report)
        report_file.close()
    log.info(f'Class report: \n {report} \n written in {report_file_path}')

    # confusion matrix
    conf_matrix_file_path = os.path.join(model_path, 'confusion_matrix.png')
    conf_matrix_fig = test_confusion_matrix(y_test_true, y_pred_argmax,
                                            t_labels, t_names)
    conf_matrix_fig.savefig(conf_matrix_file_path, bbox_inches='tight')
    log.info(f'Confusion matrix saved in {conf_matrix_file_path}')

    # samples visualization
    samples_file_path = os.path.join(model_path, 'samples.png')
    samples_fig = samples_viz(test_images, y_pred, testset_path, labels, 16)
    samples_fig.savefig(samples_file_path,  bbox_inches='tight')
    log.info(f'Sample visualization saved in {samples_file_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, dest='model_path')
    parser.add_argument('--weights', required=True, dest='weights')
    parser.add_argument('--testset_path', required=True, dest='testset_path')
    args = parser.parse_args()

    test(args.model_path, args.weights, args.testset_path)
