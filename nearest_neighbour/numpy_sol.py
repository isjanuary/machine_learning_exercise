from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


''' numpy solution without tensorflow '''
import numpy as np

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in xrange(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1) # distances 5000 * 1
      xtr_idx = np.argmin(distances) # get the index with smallest distance
      corr_ytr = self.ytr[xtr_idx] # ytr 5000 * 10, corr_ytr 1 * 10
      min_index = np.argmax(corr_ytr, 0)
      # print '####',  type(min_index), min_index
      # Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
      Ypred[i] = min_index
      print i, 'times', min_index

    return Ypred

# Xtr, Ytr = mnist.train.next_batch(10000)
# Xte, Yte = mnist.test.next_batch(2000)

tr_img = mnist.train.images
tr_labels = mnist.train.labels
te_img = mnist.test.images
te_labels = mnist.test.labels

print type(tr_img), type(te_img)
print tr_img.shape, te_img.shape


Xtr_rows = tr_img.reshape(tr_img.shape[0], 28 * 28) # Xtr_rows becomes 5000 * 784
Xte_rows = te_img.reshape(te_img.shape[0], 28 * 28) # Xte_rows becomes 200 * 784

nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(Xtr_rows, tr_labels) # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
''' Yte_predict is an array(1 * 200), each element of which is the index of 1 in 
    Yte one-hot vector '''
te_max = np.argmax(te_labels, 1)
# print te_max, te_max.shape
print 'accuracy: %f' % ( 1 - np.count_nonzero(Yte_predict - te_max) / float(te_img.shape[0]) )

