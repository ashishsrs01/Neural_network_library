import numpy as np
from tensor import Tensor


class Neuron:

    def __init__(self, nin):

        self.w = [Tensor(np.random.randn())
                   for _ in range(nin)]
        
        self.b = Tensor(0.0)

    def __call__(self, x):

        out = self.b

        for wi, xi in zip(self.w, x):
            out = out + wi * xi

        return out.relu()

    def parameters(self):
        return self.w + [self.b]