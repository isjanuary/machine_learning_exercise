from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

input_num_units = 28 * 28
hidden_num_units = 500
output_num_units = 10
seed = 128

x = tf.placeholder(tf.float32, [None, input_num_units])
y_ = tf.placeholder(tf.float32, [None, output_num_units])

hidden_weights = tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed))
bias_hidden = tf.Variable(tf.random_normal([hidden_num_units], seed=seed))
hidden_layer = tf.add(tf.matmul(x, hidden_weights), bias_hidden)
hidden_layer = tf.nn.relu(hidden_layer)

hidden_output = tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
bias_output = tf.Variable(tf.random_normal([output_num_units], seed=seed))
output_layer = tf.matmul(hidden_layer, hidden_output) + bias_output

# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
# y = tf.matmul(x, W) + b

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output_layer))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(output_layer), 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))