import pandas
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

sess = tf.Session()
df = pandas.read_csv('./admit.csv')
# print df['Sales']
# print df['TV']
# print dir(df)
admit = df['admit'].tolist()
gre = df['gre'].tolist()
rank = df['rank'].tolist()
gpa = df['gpa'].tolist()
##########
# print df.columns
# print len(admit)


# declare placeholder
x_ph = tf.placeholder(tf.float32)
y_ph = tf.placeholder(tf.float32)
# grant init value of var
b0 = tf.Variable(0.0, tf.float32)
b1 = tf.Variable(0.0, tf.float32)

# define model
# f_x = tf.add(tf.multiply(x_ph, b1), b0)
# y_predicted = 1/(1 + tf.exp(-(f_x)))
# cost = tf.reduce_mean(-(y_ph * tf.log(y_predicted) + (1 - y_ph) * tf.log(1 - y_predicted) ))
f_x = tf.add(tf.multiply(b1, x_ph), b0)
Y_model = 1 / (1 + tf.exp(-f_x))
# cost = tf.reduce_mean(
#   -(
#     tf.add(
#       tf.multiply(y_ph, tf.log(Y_model)),
#       tf.multiply(1 - y_ph, tf.log( 1 - Y_model ))
#     )
#   )
# )

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_x, labels=y_ph))
cost = tf.Print(cost, [Y_model, cost], "### ")

# now start stochastic gradient descent
# step1: normalization
# X1_norm = gre / np.amax(gre)
X1_norm = np.array(gre)
Y_norm = np.array(admit)

# step2: train model
# for different learning rate, we can get different sess.run result
train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
sess.run(tf.global_variables_initializer())
for x in range(2):
  for (x_n, y_n) in zip(X1_norm, Y_norm):
    sess.run(train_op, feed_dict={x_ph: x_n, y_ph: y_n})
b0, b1 = sess.run([b0, b1])
print b0, b1

import math
print '#########  ', 1 / ( 1 + math.exp(-(b1 * 1000 + b0)) )
# print b1, '* x +', b0
# plt.scatter( X1_norm, 1 / ( 1 + np.exp(-X1_norm) ) )
# plt.plot(X1_norm, X1_norm * b1 + b0)
# plt.show()


# TODO: tensorboard draw cost function

# draw graph of f(x) = x ^ 4
# x_4 = np.power(gpa, np.full(np.array(gpa).shape, 4.0))
# x_4 = np.power(gpa, 4.0)
# plt.scatter(gpa, x_4)
# plt.plot(gpa, x_4)
# plt.show()

