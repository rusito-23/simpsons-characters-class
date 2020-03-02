# Le Simpson
## Simpsons character classification

The main goal of this project is to achieve Simpsons character classification using a CNN written using the Keras framework.

### Model

The model is a simple Convolutional Neural Network with the following structure:

- Convolutional layer + RELU
- Dropout + Batch Normalization
- Convolutional layer + RELU
- Max Pooling layer
- Dropout + Batch Normalization
- Structure above repeated 3 times ...
- Flattening layer
- (Dense + RELU + Dropout + Batch Normalization) x 2
- Dense with SoftMax Activation

### Dataset preprocessing

In order to train the model, I used the [Kaggle Simpsons Dataset](https://www.kaggle.com/alexattia/the-simpsons-characters-dataset) from alexattia.

One of the main challenges from this dataset was the data preprocessing, as there were several labels (classes) with missing data, or some labels with two much data (like Homer Simpson). The data cleaning process can be found [here](server/notebooks/study.ipynb).

### Model prediction

The model performs with a `86.34% accuracy` on a validation set. Although it does not perform well on real life human pictures ([check it here](server/notebooks/predict.ipynb)).

### Model deployment

As the objective of this project is to learn about CNNs and the different uses that can be made of them, I proposed to setup a deployment environment using ONNX Runtime (the ONNX conversion procedure can be found [here](server/notebooks/onnx-conversion.ipynb)). The final goal is to expose an API using Flask within a Docker environment to perform the prediction. This is still WIP.
