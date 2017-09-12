import pandas
import numpy as np
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
import tensorflow as tf

sess = tf.Session()
df = pandas.read_csv('./advertising.csv')
# print df['Sales']
# print df['TV']
# print dir(df)
print df.columns

X = df['TV'].tolist()
Y = df['Sales'].tolist()
# X_np = np.array(X)
#X = np.array(X)
#Y = np.array(Y)
#print type(X)
#print dir(X)
#print X.shape
#print Y.shape
# print X_norm
# print Y_norm

#writer = tf.summary.FileWriter('./graph', sess.graph)

#getting average
########
# X_const = tf.constant(X)
# Y_const = tf.constant(Y)
# tensorX = sess.run(X_const)
# tensorY = sess.run(Y_const)
# x_avg = sess.run(tf.reduce_mean(tensorX))
# y_avg = sess.run(tf.reduce_mean(tensorY))

# # caculate numerator
# x_avg_arr = tf.fill(X_const.shape, -x_avg)
# y_avg_arr = tf.fill(Y_const.shape, -y_avg)
# # print sess.run(x_avg_arr)
# # print x_avg_arr.shape
# # print X_const.shape
# X_diff_arr = tf.reduce_sum([tensorX, x_avg_arr], 0)
# Y_diff_arr = tf.reduce_sum([tensorY, y_avg_arr], 0)
# # print sess.run(X_diff_arr)
# # print sess.run(Y_diff_arr)
# # print X_diff_arr.shape

# X_mul_Y_arr = tf.multiply(X_diff_arr, Y_diff_arr)
# # print sess.run(X_mul_Y_arr)
# numerator = tf.reduce_sum([X_mul_Y_arr])

# # caculate denominator
# X_sqr_arr = tf.multiply(X_diff_arr, X_diff_arr)
# denominator = tf.reduce_sum([X_sqr_arr])

# # get beta_hat
# # sess.run will return a numerical value, OR it's a Tensor
# beta_hat_1 = sess.run(tf.divide(numerator, denominator))

# # get Intercept
# beta_hat_0 = y_avg - beta_hat_1 * x_avg
# Y_np = beta_hat_1 * X_np + beta_hat_0
# # plt.scatter(X, Y)
# # plt.plot(X_np, Y_np)
# # plt.show()

# # Y_exp_4 = X_np ^ 4
# # print Y_exp_4

# declare placeholder
x_ph = tf.placeholder(tf.float32)
y_ph = tf.placeholder(tf.float32)
# grant init value of var
b0 = tf.Variable(0.0)
b1 = tf.Variable(0.0)
# define model
Y_model = tf.add(tf.multiply(x_ph, b1), b0)
cost = tf.pow(tf.subtract(y_ph, Y_model), 2)/2

# now start stochastic gradient descent
# step1: normalization
X_norm = X / np.amax(X)
Y_norm = Y / np.amax(Y)

# step2: train model
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
sess.run(tf.global_variables_initializer())
for x in range(25):
  for (x, y) in zip(X_norm, Y_norm):
    # print sess.run(train_op, feed_dict={x_ph: x, y_ph: y})
    sess.run(train_op, feed_dict={x_ph: x, y_ph: y})
print sess.run([b0, b1])

b1 = sess.run(b1)
b0 = sess.run(b0)
plt.plot(X_norm, X_norm * b1 + b0)
plt.scatter(X_norm, Y_norm)
plt.show()

# TODO: tensorboard draw cost function
# TODO: use tf.Print to display error
