import numpy as np
from neuronio import Neuron


class MSE(Neuron):
    def __init__(self, y, a):
        Neuron.__init__(self, [y, a])
        self.diff = None
        self.m = None

    def forward(self):
        # flatten the inputs
        y = self.nodes_input[0].value.reshape(-1, 1)
        a = self.nodes_input[1].value.reshape(-1, 1)

        self.m = self.nodes_input[0].value.shape[0]

        self.diff = y - a

        # mse
        self.value = np.mean(self.diff**2)

    def backward(self):
        self.gradients[self.nodes_input[0]] = (2 / self.m) * self.diff
        self.gradients[self.nodes_input[1]] = (-2 / self.m) * self.diff
