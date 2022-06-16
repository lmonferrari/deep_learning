from neuronio import Neuron


class Add(Neuron):
    def __init__(self, *inputs):
        Neuron.__init__(self, inputs)

    def forward(self):
        x_value = self.nodes_input[0].value
        y_value = self.nodes_input[1].value
        self.value = x_value + y_value

