import numpy as np


class Perceptron:
    def __init__(self, _learning_rate=0.01, epochs=50):
        self._learning_rate = _learning_rate
        self._epochs = epochs
        self._w = None
        self._errors = None

    def train(self, x, y):
        self._w = np.zeros(1 + x.shape[1])
        self._errors = []

        for _ in range(self._epochs):
            errors = 0
            for xi, target in zip(x, y):
                update = self._learning_rate * (target - self.predict(xi))
                self._w[1:] += update * xi
                self._w[0] += update
                errors += int(update != 0.0)
            self._errors.append(errors)
        return self

    def net_input(self, x):
        return np.dot(x, self._w[1:]) + self._w[0]

    def predict(self, x):
        return np.where(self.net_input(x) >= 0, 1, -1)

    @property
    def w(self):
        return self._w

    @property
    def errors(self):
        return self._errors
