from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

def generate_hidden_layer(input_size, output_size, input, output, layer_size, b):
  if (layer_size == 0):
    return b
  W = tf.Variable(tf.zeros([]))
  curr_z = 



mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
W = tf.Variable(
  tf.random_uniform(
    shape=[784, 10],
    minval=-1/10,
    maxval=1/10,
    dtype=tf.float32,
    seed=128
  )
)
b = tf.Variable(tf.zeros([10]))

W1 = tf.Variable(tf.zeros([784, 16]))
W2 = tf.Variable(tf.zeros([16, 10]))
b1 = tf.Variable(tf.random_normal(shape=[16], dtype=tf.float32, seed=64))
b2 = tf.Variable(tf.random_normal(shape=[10], dtype=tf.float32, seed=64))
Z2 = tf.matmul(x, W1) + b1 # Z2.shape: (784, 16)
A2 = tf.sigmoid(Z2) # A2.shape: (784, 16)
Z3 = tf.matmul(A2, W2) + b2 # Z3.shape: (10,)
A3 = tf.nn.softmax(Z3)

y = tf.matmul(x, W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=y_))
cross_entropy = tf.Print(cross_entropy, [W1, W2], "--- W1, W2")
train_step = tf.train.GradientDescentOptimizer(.5).minimize(cross_entropy)


with tf.control_dependencies([tf.Print(cross_entropy, [W1, W2], "### W1 W2", summarize=5)]):
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  # running sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) here won't
  # print anything in the console. 2 solutions to print cross_entropy here:
  # 1. cross_entropy = tf.reduce_mean(...)
  #    cross_entropy = tf.Print(cross_entropy, [cross_entropy, ...])
  #    train_step = tf.train.GradientDescentOptimizer().minimize(cross_entropy)
  #    sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
  # 2. cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_...)
  #    with tf.control_dependencies([tf.Print(cross_entropy, [cross_entropy, ...])]):
  #      train_step = tf.train.GradientDescentOptimizer().minimize(cross_entropy)
  #    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})



correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(Z3), 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_:  mnist.test.labels}))
# print "W", sess.run(W)
# print "b", sess.run(b)
# print "W1, W2", sess.run([W1, W2])
# print "b1, b2", sess.run([b1, b2])


