import matplotlib.pyplot as plt
import numpy as np

def implot(img, label):
    plt.imshow(img)
    plt.xlabel(label)
    plt.show()
    
def implot_colorbar(img, label):
    plt.imshow(img)
    plt.colorbar()
    plt.grid(False)
    plt.xlabel(label)
    plt.show()