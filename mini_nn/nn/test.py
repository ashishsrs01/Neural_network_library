from tensor import Tensor


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

x = Tensor([3.0], requires_grad=True)
print(x.data)
print(x.grad)
print(x.requires_grad)