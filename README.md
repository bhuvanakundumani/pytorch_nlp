This repository contains the implementation of text classification model - LSTM using Pytorch framework.
Text classifier is built for the IMDB Dataset. It has 50000 movie reiews classified into positive and negative. The dataset can be downlaoded from https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews. Original dataset can be got at - http://ai.stanford.edu/~amaas/data/sentiment/

Sentiment classification is done using LSTM.

data_prep.py - prepares the data 
preprocess.py - tokenizes and splits the dataset into data_valid and data_train
main.py - trains the model and has functions to predict as well.

