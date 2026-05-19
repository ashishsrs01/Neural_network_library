from tensor import Tensor

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
print(x)'''

x = Tensor(2.0)
y = Tensor(3.0)

z = x * y

print(z._prev)
print(z._op)

z.backward()
print(x)
print(y)
print(z)