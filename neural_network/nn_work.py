import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy

class NeuralNetwork():
  def __init__(self):
    #Hyper parameters --- Parameters that do not change!
    self.inputlayer_size = 2
    self.hiddenlayer_size = 3
    self.outputlayer_size = 1
    #Model parameters that we will have to learn during ML
    self.W1 = np.random.randn(self.inputlayer_size, self.hiddenlayer_size)
    self.W2 = np.random.randn(self.hiddenlayer_size, self.outputlayer_size)

  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

  def sigmoid_prime(self, x):
    return np.exp(-x)/((1 + np.exp(-x))**2)

  def costFunction(self, x, y):
    yHat = self.forward_prop(x)
    J = 0.5*sum((y - yHat)**2)
    return J

  def costFunctionPrime(self, x, y):
    self.yHat = self.forward_prop(x)

    delta3 = np.multiply(-(y - self.yHat), self.sigmoid_prime(self.z3))
    dJdW2 = np.dot(self.a2.T, delta3)

    delta2 = np.dot(delta3, self.W2.T)*self.sigmoid_prime(self.z2)
    dJdW1 = np.dot(x.T, delta2)

    return dJdW1, dJdW2

  def plot_sigmoid(self, x):
    plt.grid(True)
    plt.plot(x, self.sigmoid(x), 'ro')
    plt.show()

  def plot_sigmoid_prime(self):
    plt.grid(True)
    ip = np.arange(-5,5,0.01)
    plt.plot(ip, self.sigmoid(ip))
    plt.plot(ip, self.sigmoid_prime(ip))
    plt.show()

  def forward_prop(self, x):
    self.z2 = np.dot(x, self.W1)
    self.a2 = self.sigmoid(self.z2)
    self.z3 = np.dot(self.a2, self.W2)
    self.a3 = self.sigmoid(self.z3)
    return self.a3

  # def get_descent_direction(self, x, y):
  #   dJdW1, dJdW2 = self.costFunctionPrime(x, y)
  #   prevLost = self.costFunction(x, y)
  #   tempW1 = copy.copy(self.W1)
  #   tempW2 = copy.copy(self.W2)
  #   self.W1 += dJdW1
  #   currLost = self.costFunction(x, y)
  #   if (prevLost <= currLost):
  #     self.W1 = tempW1 - dJdW1
  #     prevLost = currLost
  #     self.W2 += dJdW2
  #     currLost = self.costFunction(x, y)
  #     if (prevLost <= currLost):
  #       self.W2 = tempW2 - dJdW2
  #   else:
  #     self.W2 += dJdW2
  #     prevLost = currLost
  #     currLost = self.costFunction(x, y)
  #     if (prevLost <= currLost):
  #       self.W2 = tempW2 - dJdW2
  #   print self.costFunction(x, y)

  def get_descent_direction(self, x, y, w1, w2, cost, stepNumber):
    if (stepNumber == 10):
      return
    prevCost = self.costFunction(x, y)
    if (prevCost > cost):
      return
    dJdW1, dJdW2 = self.costFunctionPrime(x, y)
    self.get_descent_direction(x, y, w1 + dJdW1, w2 + dJdW2, prevCost, stepNumber + 1)
    self.get_descent_direction(x, y, w1 + dJdW1, w2 - dJdW2, prevCost, stepNumber + 1)
    self.get_descent_direction(x, y, w1 - dJdW1, w2 + dJdW2, prevCost, stepNumber + 1)
    self.get_descent_direction(x, y, w1 - dJdW1, w2 - dJdW2, prevCost, stepNumber + 1)

x = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)

#Normalising:
x = x / np.amax(x)
y = y / 100.0

nn = NeuralNetwork()
# yHat = nn.forward_prop(x)
# plt.bar([0, 1, 2], y, width = 0.35, alpha=0.8)
# plt.bar([0.35, 1.35, 2.35],yHat, width = 0.35, color='r', alpha=0.8)
# plt.grid(True)
# plt.legend(['y', 'yHat'])
# plt.show()
# nn.plot_sigmoid_prime()
# d1, d2 = nn.costFunctionPrime(x, y)
# print nn.costFunction(x, y)

# nn.W1 = nn.W1 - d1
# nn.W2 = nn.W2 - d2
# print nn.costFunction(x, y)
# print nn.get_descent_direction(x, y)
import sys
# for i in range(1000):
#   nn.get_descent_direction(x, y, nn.W1, nn.W2, sys.maxint, 0)

nn.get_descent_direction(x, y, nn.W1, nn.W2, sys.maxint, 0)


print y
print nn.forward_prop(x)
