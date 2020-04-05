#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import logging
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,\
                            CSVLogger, TensorBoard
from config import IN_SHAPE
from train_utils.output_folder import create_output_folder
from data_utils.dataset import SimpsonsDataset
from model.vanilla import classification_model as cm
from log import config_logger
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config_logger()
log = logging.getLogger('simpsons')


def train(dataset_path, output_path, weights,
          epochs, batch_size, split):

    # create the autoincremental output folder
    output_path = create_output_folder(output_path)
    log.info(f'Output folder: {output_path}')

    # build dataset
    dataset = SimpsonsDataset(dataset_path, batch_size, split)
    log.info('Dataset built succesfully')

    # write the labels into the output folder
    with open(os.path.join(output_path, 'labels.txt'), 'w') as file:
        file.write('\n'.join(dataset.labels))
        file.close()
    log.info(f'Classes: {dataset.num_classes} - Labels wrote succesfully')

    # create the model
    model = cm(input_shape=IN_SHAPE, class_num=dataset.num_classes)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    log.info('Model compiled')

    if weights is not None:
        model.load_weights(weights)
        log.info(f'Weights load from {weights}')

    # callbacks
    callbacks = [
        TensorBoard(log_dir=os.path.join(output_path, 'logs'),
                    write_graph=False),
        EarlyStopping(monitor='val_accuracy', verbose=1,
                      patience=epochs/4, mode='max'),
        ModelCheckpoint(os.path.join(output_path, 'chcks', 'chck_{epoch}.h5'),
                        save_best_only=False, save_weights_only=True),
        CSVLogger(os.path.join(output_path, 'history.csv'))]

    # train
    model.fit(dataset.train_gen(),
              steps_per_epoch=dataset.train_steps,
              validation_data=dataset.val_gen(),
              validation_steps=dataset.val_steps,
              epochs=epochs,
              callbacks=callbacks,
              use_multiprocessing=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True, dest='dataset_path')
    parser.add_argument('--output_path', required=True, dest='output_path')
    parser.add_argument('--weights', default=None, dest='weights')
    parser.add_argument('--epochs', default=30, dest='epochs', type=int)
    parser.add_argument('--batch_size', default=4, dest='batch_size', type=int)
    parser.add_argument('--split', default=0.33, dest='split', type=float)
    args = parser.parse_args()

    train(args.dataset_path, args.output_path, args.weights,
          args.epochs, args.batch_size, args.split)
