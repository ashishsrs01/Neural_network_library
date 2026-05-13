import numpy as np

class Tensor:

    def __init__(self,data):
        self.data = np.array(data)
    
    def __repr__(self):
        return f"Tensor({self.data})"

    def show(self):
        print(self.data)

    def __add__(self,other):
        return Tensor(self.data + other.data)

    def __mul__(self,other):
        return Tensor(self.data * other.data)

    def shape(self):
        return (self.data.shape)








