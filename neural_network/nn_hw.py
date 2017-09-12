import numpy as np

class NeuralNetwork:
  def __init__:
    

  def costFunction(self, x, y):
    yHat = self.forward_prop(x)
    J = 0.5 * sum(y - yHat)
    return J
  
  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
  
  def sigmoidPrime(self, x):
    return np.exp(x) / ( 1 + np.exp(x) ** 2 )
    
  def costFunctionPrime(self, X, y):
    self.yHat = self.forward(X)
     delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
    dJdW2 = np.dot(self.a2.T, delta3)
     delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
    dJdW1 = np.dot(X.T, delta2
     return dJdW1, dJdW2