import numpy as np

class Tensor:

    def __init__(self,data):
        self.data = np.array(data)

    def show(self):
        print(self.data)

    def __add__(self,other):
        return Tensor(self.data + other.data)

    def shape(self):
        return (self.data.shape)








