import numpy as np


class MAE():
  '''
    Mean Absolute Error
  '''

  def __init__(self):
    # keep track of how many times we pretended that the derivative at 0 is 0
    self.count_derivative_at_0 = 0
    self.count_derivative_total = 0

  def loss(self, Y_hat, Y):
    return np.mean(np.absolute(Y - Y_hat))

  def loss_derivative(self, Y_hat, Y):
    self.count_derivative_at_0 += np.sum((Y_hat - Y) == 0)
    self.count_derivative_total += Y_hat.size

    n = Y_hat.size
    return (1*(Y_hat > Y) + -1*(Y_hat < Y)) / n


if __name__ == "__main__":
  error = MAE()
  predictions = np.array([[1.1], [0.9], [1]])
  true_values = np.array([[1], [1], [1]])

  print(error.loss_derivative(predictions, true_values))
  print(error.count_derivative_at_0, error.count_derivative_total)
