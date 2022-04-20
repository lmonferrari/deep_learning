import numpy as np


class Scaler:
    def __init__(self):
        self._x = None

    def standard_scaler(self, x):
        self._x = np.copy(x)
        self._x[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
        self._x[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()

        return self._x


