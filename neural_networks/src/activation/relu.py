import numpy as np


class ReLU():
  '''
    ReLU activation layer
  '''

  def __init__(self):
    # keep track of how many times we pretended that the derivative at 0 is 0
    self.count_derivative_at_0 = 0
    self.count_derivative_total = 0

  def forward(self, X):
    self.X = X
    return np.maximum(0, X)

  def backward(self, dL_dY, learning_rate):
    self.count_derivative_at_0 += np.sum(self.X == 0)
    self.count_derivative_total += self.X.size

    dY_dX = 1 * (self.X > 0)
    return dL_dY * dY_dX

  def __str__(self) -> str:
    return f'ReLU ({self.count_derivative_at_0}/{self.count_derivative_total} derivatives at x=0)'
