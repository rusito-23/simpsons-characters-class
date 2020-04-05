# Simpsons character classification

Using Keras with Tensorflow backend.

## Model

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

## Dataset

This repo uses the [Kaggle Simpsons Dataset](https://www.kaggle.com/alexattia/the-simpsons-characters-dataset) from alexattia.

One of the main challenges for this dataset was the data preprocessing, as there were several labels missing data, or some labels with two much data (i.e *Homer Simpson*).

The data cleaning analisis can be found [here](server/notebooks/data_cleaning.ipynb).

## Train

- download the [dataset](https://www.kaggle.com/alexattia/the-simpsons-characters-dataset).
- install the [requirements](requirements.txt)
- run the script:

```
python source/train.py \
	--dataset_path path/to/downloaded/dataset \
	--output_path /path/to/model/output
```
