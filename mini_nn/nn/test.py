from tensor import Tensor
from neuron import Neuron
from layer import Layer

'''
a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])

print(a)
print(b)

print(a + b)

print(a * b)

print(a.shape())

m = Tensor([
    [1, 2],
    [3, 4]
])

print(m)
print(m.shape())

a = Tensor([[1, 2]])
b = Tensor([[3],
            [4]])


c = a @ b
print(c)


a = Tensor([10, 20, 30])
b = Tensor([2,2,2])
print(a + b)

x = Tensor([[1,2],[3,4]])

y = Tensor([[5,6],[7,8]])

print(x + y)

a = x @ y
print(a)

x = Tensor([3.0])
print(x)


x = Tensor(2.0)
y = Tensor(3.0)

a = x * y
b = a + y

b.backward()

print("x:", x)
print("y:", y)
print("a:", a)
print("b:", b


a = Tensor(5.0)
b = Tensor(2.0)

c = a - b

c.backward()

print(a.grad)
print(b.grad)

x = Tensor(3.0)

y = x ** 2

y.backward()

print("x:", x)
print("y:", y)

pred = Tensor(4.0)
target = Tensor(2.0)

loss = (pred - target) ** 2

loss.backward()

print("loss:", loss)
print("pred grad:", pred.grad)

x = Tensor(8.0)
y = Tensor(2.0)

z = x / y

z.backward()

print("z:", z)
print("x grad:", x.grad)
print("y grad:", y.grad)

x = Tensor(-4.0)
a = Tensor(4.0)

y = x.relu()
b = a.relu()

y.backward()
b.backward()

print(y)
print(x.grad)
print(b)
print(y.grad)

x = Tensor(2.0)

y = (x * 3 + 5).relu()

y.backward()

print(y)
print(x.grad)


n = Neuron(3)

x = [2.0, 3.0, 4.0]

y = n(x)

print(y)

y.backward()

for p in n.parameters():
    print(p)'''

layer = Layer(3, 2)

x = [2.0, 3.0, 4.0]

out = layer(x)

loss = out[0] + out[1]

loss.backward()

for p in layer.parameters():
    print(p.grad)