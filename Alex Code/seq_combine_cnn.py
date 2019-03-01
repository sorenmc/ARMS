
"""Adapted from http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/"""
import tensorflow as tf
import numpy as np



class SeqCombineCNN(object):
    def __init__(
        self, sequence_length, num_classes,
        filter_lengths, num_filters, dropout_keep=1.0
    ):

        self.X = tf.placeholder(tf.float32, [None, sequence_length, 3])
        self.y = tf.placeholder(tf.float32, [None, num_classes])

        # additional dimension for "channels"
        # used in RGB for images, but we only have one channel
        X_channels = tf.expand_dims(self.X, -1)
        
        with tf.device('/cpu:0'), tf.name_scope("CNN"):
            filter_outputs = []
            for filter_length in filter_lengths:
                with tf.name_scope("filter{}".format(filter_length)):
                    W = tf.Variable(
                        tf.truncated_normal([filter_length, 3, 1, num_filters], name="W")
                    )
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                    
                    conv = tf.nn.conv2d(
                        X_channels,
                        W,
                        # sliding window does not skip any examples/steps/features/channels
                        strides=[1,1,1,1],
                        padding="VALID",
                        name="conv"
                    )

                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                    # pooled over entire sequence --> each example converted into 1 num_filters-dimensional vector
                    pooled = tf.nn.max_pool(
                        h,
                        ksize = [1, sequence_length - filter_length + 1, 1, 1],
                        strides=[1,1,1,1],
                        padding="VALID",
                        name="pool"
                    )
          
                   
                    filter_outputs.append(pooled)


            combined_filters = tf.concat(axis=3, values=filter_outputs)
            combined_filters = tf.reshape(combined_filters, [-1, num_filters * len(filter_lengths)])
            combined_filters = tf.nn.dropout(combined_filters, keep_prob=dropout_keep, name="final_dropout")

            with tf.name_scope("output"):
                # Transform pooled vectors to prediction logits
                W = tf.Variable(tf.truncated_normal([num_filters * len(filter_lengths), num_classes], stddev=0.1), name="W")
                b = tf.Variable(tf.truncated_normal([num_classes], stddev=0.1))
                self.logits = tf.nn.xw_plus_b(combined_filters, W, b, name="logits")
                self.predictions = tf.argmax(self.logits, 1, name="predictions")


            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
                self.loss = tf.reduce_mean(losses)