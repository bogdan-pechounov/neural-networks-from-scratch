class NeuralNetwork():
  '''
    Neural network divided into layers
  '''

  def __init__(self, error):
    self.layers = []
    self.error = error

  def add(self, layer):
      '''
        Add layer
      '''
      self.layers.append(layer)

  def forward(self, X):
      '''
        Propagate input forward through each layer
      '''
      for layer in self.layers:
          # output of current layer becomes input of next layer
          X = layer.forward(X)
      return X

  def backward(self, dL_dY, learning_rate):
      '''
        Propagate loss backwards 
      '''
      for layer in reversed(self.layers):
          # input derivative of current layer becomes output derivative of previous layer
          dL_dY = layer.backward(dL_dY, learning_rate)
      return dL_dY

  def train(self, X, Y, epochs=100, learning_rate=0.01):
      '''
        Perform backprogation for a given number of epochs
      '''
      for i in range(epochs):
          Y_hat = self.forward(X)
          dL_dY_hat = self.error.loss_derivative(Y_hat, Y)
          self.backward(dL_dY_hat, learning_rate)

  def __str__(self) -> str:
      lines = []
      seperator = '-----'
      inner = f'{seperator}\n'.join([str(layer) for layer in self.layers])
      return f'{seperator}\n{inner}{seperator}\n'
