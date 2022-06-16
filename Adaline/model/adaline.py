import numpy as np


class AdalineGD:

    def __init__(self, learning_rate=0.01, epochs=60):
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._cost = None
        self._w = None

    def train(self, x, y):
        self._w = np.zeros(1 + x.shape[1])
        self._cost = []

        for i in range(self._epochs):
            output = self.net_input(x)
            errors = (y - output)
            self._w[1:] += self._learning_rate * x.T.dot(errors)
            self._w[0] += self._learning_rate * errors.sum()
            cost = (errors**2).sum() / 2.0
            self._cost.append(cost)
        return self

    def net_input(self, x):
        return np.dot(x, self._w[1:]) + self._w[0]

    def activation(self, x):
        return self.net_input(x)

    def predict(self, x):
        return np.where(self.activation(x) >= 0.0, 1, -1)

    @property
    def cost(self):
        return self._cost
