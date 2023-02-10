import numpy as np


class MSE():
  @staticmethod
  def loss(Y_hat, Y):
    return np.mean(np.square(Y - Y_hat))

  @staticmethod
  def loss_derivative(Y_hat, Y):
    n = Y_hat.size
    return 2 * (Y_hat - Y) / n
