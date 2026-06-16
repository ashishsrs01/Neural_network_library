from layer import Layer


class MLP:

    def __init__(self, n_in, n_outs):

        sz = [n_in] + n_outs

        self.layers= [
                       Layer(sz[i], sz[i+1])
                       for i in range(len(n_outs))
        ]

    def __call__(self, x):

        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):

        params =[]

        for layer in self.layers:
            params.extend(layer.parameters())

        return params