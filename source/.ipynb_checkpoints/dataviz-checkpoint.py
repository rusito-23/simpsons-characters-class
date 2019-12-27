import matplotlib.pyplot as plt
import numpy as np

def implot(img, label):
    plt.imshow(img)
    plt.xlabel(label)
    plt.show()
    
def implotr(imgs, lbls, labels, r):
    plt.figure(figsize = (10, 10))
    for i in range(r):
        plt.subplot(5,5,i+1)
        plt.imshow(imgs[i], cmap=plt.cm.binary)
        plt.colorbar()
        plt.grid(False)
        plt.xlabel(labels[lbls[i]])
    plt.show()
    
def implot_colorbar(img, label):
    plt.imshow(img)
    plt.colorbar()
    plt.grid(False)
    plt.xlabel(label)
    plt.show()