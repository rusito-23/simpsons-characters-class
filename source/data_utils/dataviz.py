#!/usr/bin/env python
# coding: utf-8

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import cv2
from mpl_toolkits.axes_grid1 import AxesGrid


def implot(img, label):
    """
    Show and image with a label
    """
    plt.imshow(img)
    plt.xlabel(label)
    plt.show()


def implotr(imgs, lbls, labels, r):
    """
    Plot r images with corresponding labels
    """
    plt.figure(figsize=(10, 10))
    for i in range(r):
        plt.subplot(5, 5, i+1)
        plt.imshow(imgs[i], cmap=plt.cm.binary)
        plt.grid(False)
        plt.xlabel(labels[lbls[i]])
    plt.subplots_adjust(hspace=0.2)
    plt.show()


def implot_colorbar(img, label):
    plt.imshow(img)
    plt.grid(False)
    plt.xlabel(label)
    plt.show()


def test_confusion_matrix(y_test_true, y_pred_argmax,
                          t_labels, t_names):
    fig, ax = plt.subplots(figsize=(8, 8))
    cnf_matrix = confusion_matrix(y_test_true, y_pred_argmax, labels=t_labels)
    im = ax.imshow(cnf_matrix, interpolation='nearest')
    plt.colorbar(im)
    tick_marks = np.arange(len(t_names))
    _ = plt.xticks(tick_marks, t_names, rotation=90)
    _ = plt.yticks(tick_marks, t_names)
    return fig


def load_viz_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (352, 352))
    return image


def samples_viz(test_images, y_pred, testset_path, labels, samples):
    fig, ax = plt.subplots(figsize=(20, 20.5))
    grid = AxesGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0, label_mode="1")
    font = cv2.FONT_HERSHEY_SIMPLEX
    ax.set_xticks([])
    ax.set_yticks([])

    for i in range(samples):
        # load a random image from the test set
        im_index = np.random.randint(0, len(test_images))
        image_name = test_images[im_index]
        image_path = os.path.join(testset_path, image_name)
        image = load_viz_image(image_path)

        # get real character name
        actual = image_name.split('_')[0].title()

        # get best three predictions
        best_preds = sorted(y_pred[im_index], reverse=True)[:3]
        labels_idxs = [list(y_pred[im_index]).index(p) for p in best_preds]
        best_labels = [labels[ix].split('_')[0].title() for ix in labels_idxs]
        best_preds_text = [f'{name}: {pred*100:.1f}%'
                           for name, pred in zip(best_labels, best_preds)]

        # show real and best predictions in image
        cv2.rectangle(image, (0, 260), (215, 352),
                      (255, 255, 255), -1)
        cv2.putText(image, 'Actual : %s' % actual,
                    (10, 280), font,
                    0.7, (0, 0, 0), 2, cv2.LINE_AA)
        for k, t in enumerate(best_preds_text):
            cv2.putText(image, t, (10, 300 + k*18),
                        font, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

        grid[i].imshow(image)
        grid[i].set_xticks([])
        grid[i].set_yticks([])

    return fig
