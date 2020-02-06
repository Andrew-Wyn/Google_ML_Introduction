from __future__ import print_function
from sklearn import metrics
from IPython import display
import tensorflow as tf

import collections
import io
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_function(record):
    """Extracts features and labels.

    Args:
      record: File path to a TFRecord file    
    Returns:
      A `tuple` `(labels, features)`:
        features: A dict of tensors representing the features
        labels: A tensor with the corresponding labels.
    """
    features = {
        # terms are strings of varying lengths
        "terms": tf.VarLenFeature(dtype=tf.string),
        # labels are 0 or 1
        "labels": tf.FixedLenFeature(shape=[1], dtype=tf.float32)
    }

    parsed_features = tf.parse_single_example(record, features)

    terms = parsed_features['terms'].values
    labels = parsed_features['labels']

    return {'terms': terms}, labels


# Create an input_fn that parses the tf.Examples from the given files,
# and split them into features and targets.
def _input_fn(input_filenames, num_epochs=None, shuffle=True):

    # Same code as above; create a dataset and map features and labels.
    ds = tf.data.TFRecordDataset(input_filenames)
    ds = ds.map(_parse_function)

    if shuffle:
        ds = ds.shuffle(10000)

    # Our feature data is variable-length, so we pad and batch
    # each field of the dataset structure to whatever size is necessary.
    ds = ds.padded_batch(25, ds.output_shapes)

    ds = ds.repeat(num_epochs)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.ERROR)
  train_url = 'https://download.mlcc.google.com/mledu-datasets/sparse-data-embedding/train.tfrecord'
  train_path = tf.keras.utils.get_file(train_url.split('/')[-1], train_url)
  test_url = 'https://download.mlcc.google.com/mledu-datasets/sparse-data-embedding/test.tfrecord'
  test_path = tf.keras.utils.get_file(test_url.split('/')[-1], test_url)

  # Create the Dataset object.
  ds = tf.data.TFRecordDataset(train_path)
  # Map features and labels with the parse function.
  ds = ds.map(_parse_function)

  print(ds)

  n = ds.make_one_shot_iterator().get_next()
  sess = tf.Session()
  print(sess.run(n))

  # 50 informative terms that compose our model vocabulary
  informative_terms = ["bad", "great", "best", "worst", "fun", "beautiful",
                        "excellent", "poor", "boring", "awful", "terrible",
                        "definitely", "perfect", "liked", "worse", "waste",
                        "entertaining", "loved", "unfortunately", "amazing",
                        "enjoyed", "favorite", "horrible", "brilliant", "highly",
                        "simple", "annoying", "today", "hilarious", "enjoyable",
                        "dull", "fantastic", "poorly", "fails", "disappointing",
                        "disappointment", "not", "him", "her", "good", "time",
                        "?", ".", "!", "movie", "film", "action", "comedy",
                        "drama", "family"]


  terms_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
      key="terms", vocabulary_list=informative_terms)
    
  my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

  feature_columns = [ terms_feature_column ]


  classifier = tf.estimator.LinearClassifier(
    feature_columns=feature_columns,
    optimizer=my_optimizer,
  )

  classifier.train(
    input_fn=lambda: _input_fn([train_path]),
    steps=1000)

  # evaluate return all specifics about the model, accuracy, recall, precision, loss 
  evaluation_metrics = classifier.evaluate(
    input_fn=lambda: _input_fn([train_path]),
    steps=1000)
  print("Training set metrics:")
  for m in evaluation_metrics:
    print(m, evaluation_metrics[m])
  print("---")

  evaluation_metrics = classifier.evaluate(
    input_fn=lambda: _input_fn([test_path]),
    steps=1000)

  print("Test set metrics:")
  for m in evaluation_metrics:
    print(m, evaluation_metrics[m])
  print("---")

  # now use a deep neural network to do the same 
  ##################### Here's what we changed ##################################
  classifier = tf.estimator.DNNClassifier(                                      #
    feature_columns=[tf.feature_column.indicator_column(terms_feature_column)], #
    hidden_units=[20,20],                                                       #
    optimizer=my_optimizer,                                                     #
  )                                                                             #
  ###############################################################################

  try:
    classifier.train(
      input_fn=lambda: _input_fn([train_path]),
      steps=1000)

    evaluation_metrics = classifier.evaluate(
      input_fn=lambda: _input_fn([train_path]),
      steps=1)
    print("Training set metrics:")
    for m in evaluation_metrics:
      print(m, evaluation_metrics[m])
    print("---")

    evaluation_metrics = classifier.evaluate(
      input_fn=lambda: _input_fn([test_path]),
      steps=1)

    print("Test set metrics:")
    for m in evaluation_metrics:
      print(m, evaluation_metrics[m])
    print("---")
  except ValueError as err:
    print(err)