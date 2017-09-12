import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

''' review vision of numpy solution code '''
# get input data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
Xtr = mnist.train.images
Ytr = mnist.train.labels
Xte = mnist.test.images
Yte = mnist.test.labels
Xvalid = mnist.validation.images
Yvalid = mnist.validation.labels

# print type(Yvalid), Xvalid.shape, Yvalid.shape

def predict(Xtr, Ytr, Xte):
  loops = Xte.shape[0]
  Y_pred_idx = np.zeros(loops)
  # Y_pred_idx = np.zeros(1000)

  for i in range(loops):
    dist = np.sum(np.abs(Xtr - Xte[i, :]), 1)
    corr_ytr_idx = np.argmin(dist, 0)
    corr_ytr = Ytr[corr_ytr_idx]
    Y_pred_idx[i] = np.argmax(corr_ytr)
    print i, 'times', Y_pred_idx[i]
    
  # print len(Y_pred_idx), Y_pred_idx[0:10]
  return Y_pred_idx


Y_predict = predict(Xtr, Ytr, Xvalid)
Yvalid = np.argmax(Yvalid, 1)
print 'accuary is: ', 1 - np.count_nonzero(Y_predict - Yvalid) / float(Xvalid.shape[0])

