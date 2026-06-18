import numpy as np
from tensor import Tensor


class Neuron:

    def __init__(self, n_in, activation = True):

        self.w = [Tensor(np.random.randn())
                   for _ in range(n_in)]
        
        self.b = Tensor(0.0)
        self.activation = activation

    def __call__(self, x):

        out = self.b

        for wi, xi in zip(self.w, x):
            out = out + wi * xi

        if self.activation:
            return out.tanh()

        return out

    def parameters(self):
        return self.w + [self.b]