import numpy as np


class ReLU():
    '''
      ReLU activation layer
    '''

    def __init__(self):
        # keep track of how many times we pretended that the derivative at 0 is 0
        self.count_derivative_at_0 = 0
        self.count_derivative_total = 0

    def forward(self, input):
        self.X = input
        return np.maximum(0, input)

    def backward(self, dL_dY, learning_rate):
        self.count_derivative_at_0 += np.sum(self.X == 0)
        self.count_derivative_total += self.X.size

        dY_dX = 1 * (self.X > 0)
        print(dY_dX)
        return dL_dY * dY_dX

    def __str__(self) -> str:
        return f'ReLU ({self.count_derivative_at_0}/{self.count_derivative_total} derivatives at x=0)'


if __name__ == "__main__":
    input = np.array([[-1, 0, 0.2, 1], [0, -1, 2, 0]])
    activation = ReLU()
    output = activation.forward(input)
    print(input, output)

    activation.backward(np.ones(4), 0)
    print(activation)
