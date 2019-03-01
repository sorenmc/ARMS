"""Adapted from http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/"""
import tensorflow as tf
import numpy as np
from math import floor



class SeqDeepCNN(object):
    def __init__(
        self, sequence_length, num_classes,
        filter_lengths, num_filters, pool_window, dropout_keep=1.0
    ):

        self.X = tf.placeholder(tf.float32, [None, sequence_length, 3])
        self.y = tf.placeholder(tf.float32, [None, num_classes])

        with tf.device('/cpu:0'), tf.name_scope("CNN"):

            # Kernel for first alyer
            W = tf.Variable(
                tf.truncated_normal([filter_lengths[0], 3, 1, num_filters[0]], name="W")
            )
            b = tf.Variable(tf.constant(0.1, shape=[num_filters[0]]), name="b")

            # Kernel for additional layer
            W_2 = tf.Variable(
                tf.truncated_normal([filter_lengths[1], num_filters[0], 1, num_filters[1]], name="W_2")
            )
            b_2 = tf.Variable(tf.constant(0.1, shape=[num_filters[1]]), name="b_2")

            #First convolution/pool
            X_channels = tf.expand_dims(self.X, -1)
            conv = tf.nn.conv2d(
                X_channels,
                W,
                # sliding window does not skip any examples/steps/features/channels
                strides=[1,1,1,1],
                padding="VALID",
                name="conv"
            )

            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            print(h.shape)

            # pool together sequences of k steps
            pooled = tf.nn.max_pool(
                h,
                ksize = [1,pool_window,1,1],
                strides=[1,pool_window,1,1],
                padding="VALID",
                name="pool"
            )


            pooled_sequence_length = floor((sequence_length - filter_lengths[0] + 1) / pool_window)

            pooled = tf.reshape(pooled, [-1, pooled_sequence_length, num_filters[0], 1])
            pooled = tf.nn.dropout(pooled, keep_prob=dropout_keep, name="intermediate_dropout")


            # Second convolution/pool
            conv = tf.nn.conv2d(
                pooled,
                W_2,
                # sliding window does not skip any examples/steps/features/channels
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv_2"
            )
            h = tf.nn.relu(tf.nn.bias_add(conv, b_2), name="relu")

            # pool together enter sequence
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, pooled_sequence_length - filter_lengths[1] + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="pool"
            )

            pooled = tf.reshape(pooled, [-1, num_filters[1]])
            pooled = tf.nn.dropout(pooled, keep_prob=dropout_keep, name="final_dropout")

            with tf.name_scope("output"):
                # Transform pooled vectors to prediction logits
                W = tf.Variable(tf.truncated_normal([num_filters[1], num_classes], stddev=0.1), name="W")
                b = tf.Variable(tf.truncated_normal([num_classes], stddev=0.1))
                self.logits = tf.nn.xw_plus_b(pooled, W, b, name="logits")
                self.predictions = tf.argmax(self.logits, 1, name="predictions")


            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
                self.loss = tf.reduce_mean(losses)
