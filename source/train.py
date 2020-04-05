#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import logging
from config import IN_SHAPE
from train_utils.output_folder import create_output_folder
from data_utils.dataset import SimpsonsDataset
from model.vanilla import classification_model as cm
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,\
                            CSVLogger, TensorBoard

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('simpsons')


def train(dataset_path, output_path,
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
    log.info(f'Classes: {len(dataset.labels)} - Labels wrote succesfully')

    # create the model
    model = cm(input_shape=IN_SHAPE, class_num=len(dataset.labels))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    log.info('Model compiled')

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
    parser.add_argument('--dataset_path', dest='dataset_path')
    parser.add_argument('--output_path', dest='output_path')
    parser.add_argument('--epochs', default=30, dest='epochs')
    parser.add_argument('--batch_size', default=4, dest='batch_size', type=int)
    parser.add_argument('--split', default=0.33, dest='split', type=float)
    args = parser.parse_args()

    train(args.dataset_path, args.output_path,
          args.epochs, args.batch_size, args.split)
