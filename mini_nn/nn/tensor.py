import numpy as np

class Tensor:

    def __init__(self,data, _children = (), _op =''):
        self.data = np.array(data, dtype = float)
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor({self.data}, grad = {self.grad})"

    def show(self):
        print(self.data)

    def __add__(self,other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, (self, other), '-')
        
        def _backward():
            self.grad += out.grad
            other.grad -= out.grad
        
        out._backward = _backward
        return out

    def __mul__(self,other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out

    def __matmul__(self,other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.dot(self.data, other.data), (self, other), '@')
        
        def _backward():
            self.grad += np.dot(out.grad, other.data.T)
            other.grad += np.dot(self.data.T, out.grad)
        
        out._backward = _backward
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
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def __neg__(self):
        return self * -1

    def __pow__(self, power):
        assert isinstance(power, (int, float))
        out = Tensor(self.data ** power, (self,), f'**{power}')

        def _backward():
            self.grad += (power * (self.data ** (power - 1))) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other**-1)

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0)* out.grad
        
        out._backward = _backward
        return out
        
    def _radd_(self, other):
        return self + other

    def _rsub_(self, other):
        return Tensor(other) - self

    def _rmul_(self, other):
        return self * other

    def _rtruediv_(self, other):
        return Tensor(other) / self


    def tanh(self):
        t = np.tanh(self.data)

        out = Tensor(t, (self,), "tanh")

        def _backward():
            self.grad += (1-t**2) * out.grad

        out._backward = _backward

        return out

    def sigmoid(self):
        s = 1 / (1+ np.exp(-self.data))

        out = Tensor(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        
        return out     




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






