from tensor import Tensor


a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])

print(a)
print(b)

print(a + b)

print(a * b)

print(a.shape())

# m = Tensor([
#     [1, 2],
#     [3, 4]
# ])

# print(m)
# print(m.shape())

a = Tensor([[1, 2]])
b = Tensor([[3],
            [4]])

#c = a.matmul(b)

c = a @ b

print(c)

