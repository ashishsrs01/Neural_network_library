import numpy as np

class Tensor:

    def __init__(self,data, _children = (), _op =''):
        self.data = np.array(data, dtype = float)
        self.grad = 0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor({self.data}, grad = {self.grad})"

    def show(self):
        print(self.data)

    def __add__(self,other):
        out = Tensor(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward
        return out

    def __sub__(self, other):
        out = Tensor(self.data - other.data, (self, other), '-')
        
        def _backward():
            self.grad += out.grad
            other.grad -= out.grad
        
        out._backward = _backward
        return out

    def __mul__(self,other):
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out

    def __matmul__(self,other):
        out = Tensor(np.dot(self.data, other.data))
        return out

    def square(self):
        out = Tensor(self.data**2)
        return out

    def shape(self):
        return (self.data.shape)

    def backward(self):
        # Topological sort of computation graph
        visited = set()
        topo = []
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Backward pass
        self.grad = 1
        for node in reversed(topo):
            node._backward()





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






