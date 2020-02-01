from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

import utils


tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

if __name__ == "__main__":
    california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

    california_housing_dataframe = california_housing_dataframe.reindex(
        np.random.permutation(california_housing_dataframe.index))

    # Choose the first 12000 (out of 17000) examples for training.
    training_examples = preprocess_features(california_housing_dataframe.head(12000))
    training_targets = preprocess_targets(california_housing_dataframe.head(12000))

    # Choose the last 5000 (out of 17000) examples for validation.
    validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
    validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

    # Double-check that we've done the right thing.
    print("Training examples summary:")
    display.display(training_examples.describe())
    print("Validation examples summary:")
    display.display(validation_examples.describe())

    print("Training targets summary:")
    display.display(training_targets.describe())
    print("Validation targets summary:")
    display.display(validation_targets.describe())


    # plot correlation between features and label "median house value"
    correlation_dataframe = training_examples.copy()
    correlation_dataframe["target"] = training_targets["median_house_value"]

    correlation_dataframe.corr()


    # select feature based results of correlation table plotted above
    minimal_features = [
                    'latitude',
                    'median_income'
    ]

    assert minimal_features, "You must select at least one feature!"

    minimal_training_examples = training_examples[minimal_features]
    minimal_validation_examples = validation_examples[minimal_features]

    train_model(
        learning_rate=0.001,
        steps=500,
        batch_size=5,
        training_examples=minimal_training_examples,
        training_targets=training_targets,
        validation_examples=minimal_validation_examples,
        validation_targets=validation_targets)
    
    # difficoult exercise :
    # Try creating some synthetic features that do a better job with latitude.
    # canonical use of One-Hot Encoding, with best-practice "BINNING"
    def select_and_transform_features(source_df):
        LATITUDE_RANGES = zip(range(32, 44), range(33, 45))
        selected_examples = pd.DataFrame()
        selected_examples["median_income"] = source_df["median_income"]
        for r in LATITUDE_RANGES:
            selected_examples["latitude_%d_to_%d" % r] = source_df["latitude"].apply(
            lambda l: 1.0 if l >= r[0] and l < r[1] else 0.0)
        return selected_examples

        selected_training_examples = select_and_transform_features(training_examples)
        selected_validation_examples = select_and_transform_features(validation_examples)
        
    
    _ = train_model(
        learning_rate=0.01,
        steps=500,
        batch_size=5,
        training_examples=selected_training_examples,
        training_targets=training_targets,
        validation_examples=selected_validation_examples,
        validation_targets=validation_targets
    )