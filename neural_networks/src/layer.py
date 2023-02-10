import numpy as np


class Layer():
    '''
      Fullly connected layer
    '''

    def __init__(self, input_size, output_size):
        # initialize weights and biases
        self.W = np.ones((input_size, output_size))
        self.b = np.zeros((1, output_size))

    def forward(self, X):
        self.X = X  # store input to use it during backpropagation
        return np.dot(X, self.W) + self.b

    def backward(self, dL_dY, learning_rate):
        # input derivative to be used by the previous layer
        dL_dX = np.dot(dL_dY, self.W.T)

        # weights and biases derivatives
        dL_dW_transpose = np.dot(self.X.T, dL_dY)
        dL_dB = dL_dY

        # update weights and biases
        self.W -= learning_rate * dL_dW_transpose
        self.b -= learning_rate * np.sum(dL_dB, axis=0, keepdims=True)

        return dL_dX

    def __str__(self) -> str:
        return f'Layer {self.W.shape}'
