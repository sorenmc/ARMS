"""Adapted from http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/"""
import math
import os

import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics.classification import accuracy_score

from seq_deep_cnn import SeqDeepCNN
from load_data import load_data_subset, pad_examples, convert_labels
from length_threshold import get_examples_below_length_threshold
from statistics import mean

# Test data
eval_test_performance = True

# Experiment parameters
num_folds = 10
num_epochs = 500
batch_size = 10
length_threshold = 0.95

# Model parameters
filter_length = [30, 15]
num_filters = [500, 300]
pool_window = 10

dropout_keep_prob = 0.7

# Cross validation model tracking
crossval_models = []
crossval_accuracies = []
crossval_training_accuracies = []

"""Helper functions for getting epochs & batches"""


def shuffle_for_epoch(X, y):
    size = X.shape[0]
    shuffle_indicies = np.random.permutation(size)
    return X[shuffle_indicies], y[shuffle_indicies]


def get_batch(X, y, batch_num, batch_size):
    start = batch_num * batch_size
    end = (batch_num + 1) * batch_size

    if end >= X.shape[0]:
        return X[start:], y[start:]
    else:
        return X[start:end], y[start:end]


"""Get Data"""
examples, labels = load_data_subset(indices_path="Data/train_indices.csv")
length_threshold, examples, labels = get_examples_below_length_threshold(examples, labels, threshold=length_threshold)

X, X_masks = pad_examples(examples)
y = convert_labels(labels, sorted(list(set(labels))))

"""Split off validation set"""
sss = StratifiedShuffleSplit(1, train_size=0.8)
for train_index, val_idex in sss.split(X=X, y=y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_val = X[val_idex]
    y_val = y[val_idex]

for fold in range(num_folds):
    """Split off cross validation fold"""
    sss = StratifiedShuffleSplit(1, train_size=0.8)
    for train_index, crossval_index in sss.split(X=X_train, y=y_train):
        X_trainfold = X_train[train_index]
        y_trainfold = y_train[train_index]
        X_valfold = X_train[crossval_index]
        y_valfold = y_train[crossval_index]

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
        )
        sess = tf.Session(config=session_conf)
        with sess.as_default():

            seq_cnn = SeqDeepCNN(
                sequence_length=X.shape[1],
                num_classes=y.shape[1],
                filter_lengths=filter_length,
                num_filters=num_filters,
                pool_window=pool_window,
                dropout_keep=dropout_keep_prob
            )

            """ Configure optimizer """
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-4)
            gradients = optimizer.compute_gradients(seq_cnn.loss)
            train_op = optimizer.apply_gradients(gradients, global_step=global_step)

            """ Initialize """
            sess.run(tf.initialize_all_variables())

            """ Set up model saving, model for fold 1 saved in Models/1/ """
            checkpoint_dir = os.path.abspath(os.path.join("Models", str(fold)))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")

            """ Training procedure """
            num_batches = math.floor(X_trainfold.shape[0] / batch_size)
            for epoch in range(num_epochs):
                # at every epoch, shuffle the data
                X_shuff, y_shuff = shuffle_for_epoch(X_trainfold, y_trainfold)
                for b in range(num_batches):
                    x_batch, y_batch = get_batch(X_shuff, y_shuff, batch_num=b, batch_size=batch_size)

                    feed_dict = {
                        seq_cnn.X: x_batch,
                        seq_cnn.y: y_batch
                    }

                    _, loss, step = sess.run(
                        [train_op, seq_cnn.loss, global_step],
                        feed_dict=feed_dict
                    )

                    print("fold: {} step: {}, loss {:g}".format(fold, step, loss))

            """ Evaluate on cross-validation fold """
            feed_dict = {
                seq_cnn.X: X_valfold,
                seq_cnn.y: y_valfold
            }

            predictions = sess.run(seq_cnn.predictions, feed_dict=feed_dict)
            # have to convert y_val back from one_hot into class number
            y_val_classes = np.array([np.argmax(y_val_i) for y_val_i in y_valfold])
            accuracy = accuracy_score(y_true=y_val_classes, y_pred=predictions)

            """ Get training accuracy for diagnostic purposes"""
            training_feed_dict = {
                seq_cnn.X: X_trainfold,
                seq_cnn.y: y_trainfold
            }
            training_predictions = sess.run(seq_cnn.predictions, feed_dict=training_feed_dict)
            y_train_classes = np.array([np.argmax(y_train_i) for y_train_i in y_trainfold])
            training_accuracy = accuracy_score(y_true=y_train_classes, y_pred=training_predictions)

            # Save model 'checkpoint' (all weights) in a temp directory
            model_path = saver.save(sess, checkpoint_prefix)
            crossval_models.append(model_path)

            # Record accuracy on current fold for model comparison
            crossval_accuracies.append(accuracy)
            crossval_training_accuracies.append(training_accuracy)
            print("Accuracy for fold {}: {}".format(fold, accuracy))

print('\nAccuracies\n')
for acc in crossval_accuracies:
    print(acc)

print('Mean crossval accuracy: {}'.format(mean(crossval_accuracies)))

best_model_ID = crossval_accuracies.index(max(crossval_accuracies))
print('Best performing model: {}'.format(best_model_ID))
print('With training accuracy: {}\n'.format(crossval_training_accuracies[best_model_ID]))

""" Evaluate best performing model on validation set """
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        seq_cnn = SeqDeepCNN(
            sequence_length=X.shape[1],
            num_classes=y.shape[1],
            filter_lengths=filter_length,
            num_filters=num_filters,
            pool_window=pool_window
        )


        saver = tf.train.Saver(tf.all_variables())
        saver.restore(sess, crossval_models[best_model_ID])

        feed_dict = {
            seq_cnn.X: X_val,
            seq_cnn.y: y_val
        }

        predictions = sess.run(seq_cnn.predictions, feed_dict=feed_dict)
        # have to convert y_val back from one_hot into class number
        y_val_classes = np.array([np.argmax(y_val_i) for y_val_i in y_val])
        accuracy = accuracy_score(y_true=y_val_classes, y_pred=predictions)

print("Final accuracy: {}".format(accuracy))

if not eval_test_performance:
    exit()

""" Evaluate best performing model on test set """

print("Running test data")

# Get test data
examples_test, labels_test = load_data_subset(indices_path="Data/test_indices.csv")
X_test, _ = pad_examples(examples_test)

# Test examples have to be truncated to length calculated from training set
X_test = X_test[:, :X.shape[1]]

print(X_test.shape)

y_test = convert_labels(labels_test, sorted(list(set(labels_test))))

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        seq_cnn = SeqDeepCNN(
            sequence_length=X_test.shape[1],
            num_classes=y_test.shape[1],
            filter_lengths=filter_length,
            num_filters=num_filters,
            pool_window=pool_window
        )


        saver = tf.train.Saver(tf.all_variables())
        saver.restore(sess, crossval_models[best_model_ID])

        feed_dict = {
            seq_cnn.X: X_test,
            seq_cnn.y: y_test
        }

        predictions = sess.run(seq_cnn.predictions, feed_dict=feed_dict)
        # have to convert y_val back from one_hot into class number
        y_test_classes = np.array([np.argmax(y_test_i) for y_test_i in y_test])
        accuracy = accuracy_score(y_true=y_test_classes, y_pred=predictions)

print("Final test data accuracy: {}".format(accuracy))
