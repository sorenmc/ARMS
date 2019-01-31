"""Adapted from http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/"""
import math
import os

import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics.classification import accuracy_score

from seq_cnn import SeqCNN
from load_data import load_data_subset, pad_examples, convert_labels


# Global experiment parameters
num_folds = 1
num_epochs = 200
batch_size = 10

# Cross validation model tracking
crossval_models = []
crossval_accuracies = []


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


examples, labels = load_data_subset(indices_path="Data/train_indices.csv")
X, X_masks = pad_examples(examples)
y = convert_labels(labels, list(set(labels)))

for fold in range(num_folds):

    sss = StratifiedShuffleSplit(1, train_size=0.8)
    for train_index, crossval_index in sss.split(X=X, y=y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[crossval_index]
        y_val = y[crossval_index]

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement = True,
        )
        sess = tf.Session(config=session_conf)
        with sess.as_default():

            seq_cnn = SeqCNN(
                sequence_length=X.shape[1],
                num_classes = y.shape[1],
                filter_length=20,
                num_filters=300
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
            num_batches = math.floor(X_train.shape[0] / batch_size)
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
                    break

            """ Evaluate on cross-validation fold """
            feed_dict = {
                seq_cnn.X: X_val,
                seq_cnn.y: y_val
            }

            predictions = sess.run(seq_cnn.predictions, feed_dict=feed_dict)
            # have to convert y_val back from one_hot into class number
            y_val_classes = np.array([np.argmax(y_val_i) for y_val_i in y_val])
            accuracy = accuracy_score(y_true=y_val_classes, y_pred=predictions)

            # Save model & accuracy on current fold
            crossval_accuracies.append(accuracy)
            model_path = saver.save(sess, checkpoint_prefix)
            crossval_models.append(model_path)

            print("Accuracy for fold {}: {}".format(fold, accuracy))


print('\nAccuracies\n')
for acc in crossval_accuracies:
    print(acc)

