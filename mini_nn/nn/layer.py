from neuron import Neuron 


class Layer:

    def __init__(self, n_in, n_out, activation = True):

        self.neurons = [Neuron(n_in)
                        for _ in range(n_out)]

    def __call__(self, x):

        outs = [n(x) for n in self.neurons]

        return outs[0] if len(outs) == 1 else outs

    def parameters(self):

        params= []

        for neuron in self.neurons:
            params.extend(neuron.parameters())

        return params