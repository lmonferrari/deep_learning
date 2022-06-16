import numpy as np
from neuronio import Neuron


class Linear(Neuron):
    def __init__(self, X, W, b):
        Neuron.__init__(self, [X, W, b])

    def forward(self):
        X = self.nodes_input[0].value
        W = self.nodes_input[1].value
        b = self.nodes_input[2].value

        self.value = np.dot(X, W) + b

    def backward(self):
        """
        Calculate the gradient of the output
        """
        self.gradients = {n: np.zeros_like(n.value) for n in self.nodes_input}

        for n in self.nodes_output:
            # obtaining parcial for each input node
            grad_cost = n.gradients[self]

            # partial loss in function of the inputs of this node
            self.gradients[self.nodes_input[0]] += np.dot(grad_cost, self.nodes_input[1].value.T)

            # partial loss in function of the weighs
            self.gradients[self.nodes_input[1]] += np.dot(self.nodes_input[0].value.T, grad_cost)

            # partial loss in function of the bias
            self.gradients[self.nodes_input[2]] += np.sum(grad_cost, axis=0, keepdims=False)
