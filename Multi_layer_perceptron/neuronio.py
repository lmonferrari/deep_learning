

class Neuron:
    def __init__(self, nodes_input=[]):
        # Nó(s) a partir do qual este Nó recebe valores
        self.nodes_input = nodes_input
        # Nó(s) ao qual este Nó passa valores
        self.nodes_output = []
        # valor calculado
        self.value = None
        self.gradients = {}
        # Para cada nó de entrada aqui, adicione este Nó como um Nó de saída para o Node.
        for n in self.nodes_input:
            n.nodes_output.append(self)

    def forward(self):
        raise NotImplemented

    def backward(self):
        raise NotImplemented

