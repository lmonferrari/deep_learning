import numpy as np

from neuronio import Neuron


class Sigmoid(Neuron):
    def __init__(self, node):
        Neuron.__init__(self, [node])

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def forward(self):
        input_value = self.nodes_input[0].value
        self.value = self._sigmoid(input_value)

    def backward(self):
        """
        Calculate gradient using the derivative of sigmoid function
        """
        self.gradients = {n: np.zeros_like(n.value) for n in self.nodes_input}

        for n in self.nodes_output:
            grad_cost = n.gradients[self]
            sigmoid = self.value
            self.gradients[self.nodes_input[0]] += sigmoid * (1 - sigmoid) * grad_cost
