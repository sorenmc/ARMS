"""Adapted from http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/"""
import math
import os

import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics.classification import accuracy_score

from seq_cnn import SeqCNN
from load_data import load_data_subset, pad_examples, convert_labels
from length_threshold import get_examples_below_length_threshold 
from statistics import mean

# Test data
eval_test_performance = True

# Experiment parameters
num_folds = 10
num_epochs = 200
batch_size = 10

# Model parameters
filter_length = 50
num_filters = 2000

# Cross validation model tracking
crossval_models = []
crossval_accuracies = []

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
_, examples, labels = get_examples_below_length_threshold(examples, labels, threshold=0.9)
X, X_masks = pad_examples(examples)
y = convert_labels(labels, list(set(labels)))

print(X.shape)

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
            allow_soft_placement = True,
        )
        sess = tf.Session(config=session_conf)
        with sess.as_default():

            seq_cnn = SeqCNN(
                sequence_length=X.shape[1],
                num_classes = y.shape[1],
                filter_length=filter_length,
                num_filters=num_filters
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
                X_shuff, y_shuff = shuffle_for_epoch(X, y)
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

                    print("step: {}, loss {:g}".format(step, loss))

            """ Evaluate on cross-validation fold """
            feed_dict = {
                seq_cnn.X: X_valfold,
                seq_cnn.y: y_valfold
            }

            predictions = sess.run(seq_cnn.predictions, feed_dict=feed_dict)
            # have to convert y_val back from one_hot into class number
            y_val_classes = np.array([np.argmax(y_val_i) for y_val_i in y_valfold])
            accuracy = accuracy_score(y_true=y_val_classes, y_pred=predictions)

            # Save model 'checkpoint' (all weights) in a temp directory
            model_path = saver.save(sess, checkpoint_prefix)
            crossval_models.append(model_path)

            # Record accuracy on current fold for model comparison
            crossval_accuracies.append(accuracy)
            print("Accuracy for fold {}: {}".format(fold, accuracy))


print('\nAccuracies\n')
for acc in crossval_accuracies:
    print(acc)

print('Mean crossval accuracy: {}'.format(mean(crossval_accuracies)))

best_model_ID = crossval_accuracies.index(max(crossval_accuracies))
print('Best performing model: {}'.format(best_model_ID))

""" Evaluate best performing model on validation set """
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        seq_cnn = SeqCNN(
            sequence_length=X.shape[1],
            num_classes=y.shape[1],
            filter_length=filter_length,
            num_filters=num_filters
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




