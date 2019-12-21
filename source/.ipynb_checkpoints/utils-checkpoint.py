import matplotlib.pyplot as plt
import numpy as np

def implot(img, label):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img, cmap=plt.cm.binary)
    plt.xlabel(label)
    
def implot_colorbar(img):
    plt.imshow(img)
    plt.colorbar()
    plt.grid(False)
    
def plotpredict(predictions, test_labels, test_images, labels, _range=25):
    for i in range(_range):
        pred = labels[np.argmax(predictions[i])]
        exp = labels[test_labels[i]]
        plt.figure(figsize=(12, 12))
        implot(test_images[i], "pred: {} - exp: {}".format(pred, exp))
        plt.show()