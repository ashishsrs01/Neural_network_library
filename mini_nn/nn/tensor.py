import numpy as np

class Tensor:

    def __init__(self,data, requires_grad = False):
        self.data = np.array(data, dtype = float)
        self.requires_grad = requires_grad
        self.grad = None

    def __repr__(self):
        return f"Tensor({self.data}, grad = {self.grad})"

    def show(self):
        print(self.data)

    def __add__(self,other):
        return Tensor(self.data + other.data)

    def __sub__(self, other):
        return Tensor(self.data - other.data)

    def __mul__(self,other):
        return Tensor(self.data * other.data)

    def __matmul__(self,other):
        return Tensor(np.dot(self.data, other.data))

    def square(self):
        return Tensor(self.data**2)

    def shape(self):
        return (self.data.shape)





# x = Tensor([1,2,3,4])
# y = Tensor([1,2,3,4])
# x.show()
# y.show()
# c = x + y
# print(c)
# c.show()

# x = Tensor([10,20,30])
# y = Tensor([40,50,60])
# z = Tensor([70,80,90])
# p = Tensor([[1,2,3],
#             [4,5,6]])
# x.show()
# y.show()
# z.show()

# a = p.shape()
# print(a)






