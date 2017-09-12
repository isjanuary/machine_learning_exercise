from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

''' tensorflow solutions '''
x_tr_ph = tf.placeholder(dtype=tf.float32, shape=[None, 784])
x_te_ph = tf.placeholder(dtype=tf.float32, shape=[784])

Xtr = mnist.train.images
Ytr = mnist.train.labels
Xte = mnist.test.images
Yte = mnist.test.labels

sess = tf.Session()

class NearestNeighbour():
  def __init__(self):
    pass

  def fill(self, xtr, ytr):
    self.Xtr = xtr
    self.Ytr = ytr

  def predict(self, Xte, Yte):
    accuracy = 0.
    test_cnt = Xte.shape[0]
    dist = tf.reduce_sum(tf.abs(x_tr_ph - x_te_ph), 1)
    train_op = tf.argmin(dist, 0)
    # train_op = tf.Print(train_op, [dist])
    # for i in range(100):
    for i in range(test_cnt):
      corr_ytr_idx = sess.run(train_op, feed_dict={x_tr_ph: self.Xtr, x_te_ph: Xte[i,:]})
      corr_ytr = self.Ytr[corr_ytr_idx]
      print '#####', i, 'times'
      # print 'test data:', Yte[i], 'predict data:', corr_ytr
      if (np.argmax(corr_ytr, 0) == np.argmax(Yte[i])):
        accuracy += 1. / test_cnt

    return accuracy


nn = NearestNeighbour()
nn.fill(Xtr, Ytr)
accuracy = nn.predict(Xte, Yte)
print 'accuracy is: ', accuracy

