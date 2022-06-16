import numpy as np


class AdalineSGD:

    def __init__(self, learning_rate=0.001, epochs=10):
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._w = None
        self._costs = None

    def train(self, x, y, reinitialize_weights=True):

        if reinitialize_weights:
            self._w = np.zeros(1 + x.shape[1])

        self._costs = []

        for i in range(self._epochs):
            for xi, target, in zip(x, y):
                output = self.net_input(xi)
                error = (target - output)
                self._w[1:] += self._learning_rate * xi.dot(error)
                self._w[0] += self._learning_rate * error

            cost = ((y - self.activation(x))**2).sum() / 2.0
            self._costs.append(cost)

        return self

    def net_input(self, x):
        return np.dot(x, self._w[1:]) + self._w[0]

    def activation(self, x):
        return self.net_input(x)

    def predict(self, x):
        return np.where(self.activation(x) >= 0.0, 1, -1)

    @property
    def cost(self):
        return self._costs
