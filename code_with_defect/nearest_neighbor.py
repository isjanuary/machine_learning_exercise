from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.Session()
Xtr, Ytr = mnist.train.next_batch(5000)

# distance = tf.reduce_sum(tf.abs(Xtr - x_test))

x_tr, y_tr = mnist.train.next_batch(1000)
# y_tr = mnist.train.labels

# print mnist.train.images.shape
# print mnist.train.labels.shape
# # print mnist.train.images
# # print len(x_tr[0])

x_test, y_test = mnist.test.next_batch(1000)
# y_test = mnist.test.labels

def cal_dist(x_tr, x_test):
  loops = len(x_tr)
  print 'length', loops
  dist_arr = []
  for i in range(loops):
    dist = tf.reduce_sum(tf.abs(x_tr[i] - x_test))
    dist_arr.append(sess.run(dist))
  return dist_arr

accuracy = 0.0
for i in range(20):
  arr = cal_dist(x_tr, x_test[i])
  closest_tr_idx = sess.run(tf.argmin(arr, 0))
  y_closest_tr = y_tr[closest_tr_idx]
  y_closest_te = y_test[i]
  if (tf.argmax(y_closest_te) == tf.argmax(y_closest_tr)):
    accuracy += 1./ 20.

# print '****', arr
# print closest_tr_idx
# print '----', y_closest_te
# print '****', y_closest_tr
print accuracy
# print '****', sess.run(compare)
