#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten,\
                         BatchNormalization, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.constraints import max_norm


def classification_model(input_shape, class_num):
    model = Sequential()

    # Convolutional layer
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))

    # Dropout 2% of the data to avoid overfitting
    model.add(Dropout(0.2))

    # Normalize inputs for the next layer
    model.add(BatchNormalization())

    # Second convolutional layer (complex representations)
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    # Learn relevant patterns
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    # Repeat
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    # Flatten result
    model.add(Flatten())
    model.add(Dropout(0.2))

    # Dense layers
    model.add(Dense(256, kernel_constraint=max_norm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(128, kernel_constraint=max_norm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    # Prediction layers
    model.add(Dense(class_num))
    model.add(Activation('softmax'))

    return model
