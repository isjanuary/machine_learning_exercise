from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics



# def build_nn(x, hidden_layers, shape, activate_func):
#   W_arr = []
#   for i in range(hidden_layers):
#     W = tf.Variable(tf.zeros(shape))
#     W_arr.append(W)
#     Z = tf.matmul(x, W)
#     A = activate_func(Z)
#     W = A
#   return A, W_arr


def build_graph(x, prev_act, num_nuerons, input_size, num_layers, user_input_num_layers, b):
  if num_layers == user_input_num_layers:
    return prev_act + b
  if num_layers == 0:
    curr_w = tf.Variable(tf.zeros([input_size, num_nuerons]))
    curr_z = tf.matmul(x, curr_w)
    tf.Print(curr_z, [W, b])
  else:
    curr_w = tf.Variable(tf.zeros([num_nuerons, num_nuerons]))
    curr_z = tf.matmul(prev_act, curr_w)
    tf.Print(curr_z, [W, b])
  curr_act = tf.nn.sigmoid(curr_z)
  return build_graph(x, curr_act, num_nuerons, input_size, num_layers + 1, user_input_num_layers, b)




# call build_graph
# result = build_graph(x, None, 10, 784, 0, 10, b)




# # Step1: test normal distribution
# batch = mnist.train.next_batch(700)
# print 'type, shape', type (batch), len(batch)
# x_batch = batch[0]
# y_batch = batch[1]
# # np.reshape(y_batch, (700,))
# print 'type, x.shape, y.shape', x_batch.shape, y_batch.shape
# # print np.take(x_batch, 0)
# # x0_cnt = mnist.train.next_batch(1)[0][0]
# # x_cnt_rnd = [round(i, 2) for i in x0_cnt]
# # plt.hist(x_cnt_rnd)
# # plt.show()



# # Step2: test naive, independent
# # skip Step2 now cause we know in this detect number case, input data is not naive



# # apply naive Bayes model to this detect number case
# # your_data = datasets.load_iris()
# classifier = GaussianNB()
# x = mnist.train.images.tolist()
# y = [i.tolist().index(1) for i in mnist.train.labels]
# classifier.fit(x, y)
# score = metrics.accuracy_score(y, classifier.predict(x))
# print "Accuracy:", score



from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=1))
cross_entropy = tf.Print(cross_entropy, [cross_entropy], "###")
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


