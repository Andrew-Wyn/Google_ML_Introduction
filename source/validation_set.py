from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
%tensorflow_version 1.x
import tensorflow as tf
from tensorflow.python.data import Dataset

import utils


tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


if __name__ == "__main__":

    california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

    # to avoid patologicals behaviors
    california_housing_dataframe = california_housing_dataframe.reindex(
            np.random.permutation(california_housing_dataframe.index)
        )

    # TRAINING-SET
    training_examples = preprocess_features(california_housing_dataframe.head(12000))
    training_examples.describe()

    training_targets = preprocess_targets(california_housing_dataframe.head(12000))
    training_targets.describe()

    # VALIDATION-SET
    validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
    validation_examples.describe()

    validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))
    validation_targets.describe()

    # plot differences between validation set against training set

    plt.figure(figsize=(13, 8))

    ax = plt.subplot(1, 2, 1)
    ax.set_title("Validation Data")

    ax.set_autoscaley_on(False)
    ax.set_ylim([32, 43])
    ax.set_autoscalex_on(False)
    ax.set_xlim([-126, -112])
    plt.scatter(validation_examples["longitude"],
                validation_examples["latitude"],
                cmap="coolwarm",
                c=validation_targets["median_house_value"] / validation_targets["median_house_value"].max())

    ax = plt.subplot(1,2,2)
    ax.set_title("Training Data")

    ax.set_autoscaley_on(False)
    ax.set_ylim([32, 43])
    ax.set_autoscalex_on(False)
    ax.set_xlim([-126, -112])
    plt.scatter(training_examples["longitude"],
                training_examples["latitude"],
                cmap="coolwarm",
                c=training_targets["median_house_value"] / training_targets["median_house_value"].max())
    _ = plt.plot()

    # -------


    # train model, using training-set, and plot differences against validations-set
    linear_regressor = train_model(
        learning_rate=0.00003,
        steps=500,
        batch_size=5,
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets)

    # Evaluate on Test Data
    california_housing_test_data = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv", sep=",")

    test_examples = preprocess_features(california_housing_test_data)
    test_targets = preprocess_targets(california_housing_test_data)

    predict_test_input_fn = lambda: my_input_fn(
        test_examples, 
        test_targets["median_house_value"], 
        num_epochs=1, 
        shuffle=False)

    test_predictions = linear_regressor.predict(input_fn=predict_test_input_fn)
    test_predictions = np.array([item['predictions'][0] for item in test_predictions])

    root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(test_predictions, test_targets))

    print("Final RMSE (on test data): %0.2f" % root_mean_squared_error)