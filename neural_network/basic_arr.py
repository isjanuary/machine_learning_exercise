import math
import numpy as np

# x1 = np.array([0.3, 1.0])
# x2 = np.array([0.5, 0.2])
# x3 = np.array([1.0, 0.4])
x1 = np.array([1., 1.])
x2 = np.array([1., 1.])
x3 = np.array([1., 1.])
X = np.array([x1, x2, x3])

W1 = np.full((2, 3), 1.0)
# dot is for matrix multiplication
Z1 = np.dot(X, W1)
print 'Z1, ', Z1
A = 1 / (1 + np.exp(-Z1))
W2 = np.full((3, 1), 1.0)
Z2 = np.dot(A, W2)
A2 = 1 / (1 + np.exp(-Z2))
print A2


