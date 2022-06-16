from neuronio import Neuron


class Input(Neuron):
    def __init__(self):
        # Um nó de entrada não possui nós de entrada
        # Então não é necessário passar nada para o instanciador de classe Node.
        Neuron.__init__(self)

    def forward(self, value=None):
        pass

    def backward(self):
        self.gradients = {self: 0}

        for n in self.nodes_output:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost

