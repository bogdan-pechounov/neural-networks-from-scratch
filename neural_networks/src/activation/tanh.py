import numpy as np


class Tanh():
  '''
    Tanh activation layer
  '''

  def forward(self, X):
    self.X = X
    return np.tanh(X)

  def backward(self, dL_dY, learning_rate):
    dY_dX = 1 - np.square(np.tanh(self.X))
    return dL_dY * dY_dX

  def __str__(self) -> str:
    return 'Tanh'
