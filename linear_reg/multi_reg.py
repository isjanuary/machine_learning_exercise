import pandas
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

sess = tf.Session()
df = pandas.read_csv('./advertising.csv')
# print df.columns

var_tv = df['TV'].tolist()
var_radio = df['Radio'].tolist()
f_sales = df['Sales'].tolist()

tv_ph = tf.placeholder(tf.float32)
radio_ph = tf.placeholder(tf.float32)
sales_ph = tf.placeholder(tf.float32)
ph_vec = [tv_ph, radio_ph]

b0 = tf.Variable(0.0)
b_shape = sess.run(tf.fill([1, 2], 0.0)).shape
print 'b_shape: ', sess.run(b_shape)[0].shape
# b1 = tf.Variable(0.0)
# b2 = tf.Variable(0.0)
b_vec = tf.Variable(0.0, expected_shape=b_shape)

# Y_model = tf.add_n([
#   tf.multiply(tv_ph, b1),
#   tf.multiply(radio_ph, b2),
#   b0
# ])

# Y_model using matrix multiply: matmul
Y_model = tf.matmul(ph_vec, b_vec) + b0
cost = tf.pow(tf.subtract(sales_ph, Y_model), 2)/ 2

# why type of tv_norm is tf.float64, not tf.float32 here
tv_norm = var_tv / np.amax(var_tv)
radio_norm = var_radio / np.amax(var_radio)
sales_norm = f_sales / np.amax(f_sales)

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
sess.run(tf.global_variables_initializer())
for i in range(200):
  for (x1, x2, y) in zip(tv_norm, radio_norm, sales_norm):
    sess.run(train_op, feed_dict={tv_ph: x1, radio_ph: x2, sales_ph: y})

b1 = sess.run(tf.cast(b1, tf.float64))
b2 = sess.run(tf.cast(b2, tf.float64))
b0 = sess.run(tf.cast(b0, tf.float64))
# b1 = sess.run(b1)
# b2 = sess.run(b2)
# b0 = sess.run(b0)
# print 'b0 ', b0, 'type (b1) ', type (b1), 'type (tv_norm[0])', type (tv_norm[0])


# draw 3D graph starting
print '####################   draw 3D graph starting   ####################'
# initialize an axis3D instance
fig = plt.figure()
ax = fig.gca(projection='3d')

# use X, Y = np.meshgrid(X, Y) to generate 2D grid according to X, Y coordinates
# use np.math_func from 
#   https://docs.scipy.org/doc/numpy-1.12.0/reference/routines.math.html
# to calculate the Z coordinate
tv_norm, radio_norm = np.meshgrid(tv_norm, radio_norm)
Y_hat = np.add(tv_norm * b1, radio_norm * b2) + b0

ax.plot_surface(tv_norm, radio_norm, Y_hat, color='#87aeed')

plt.show()
