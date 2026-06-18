from tensor import Tensor
from mlp import MLP
from optimizer import SGD

model = MLP(1, [1])

optimizer = SGD(model.parameters(), lr = 0.01)

xs = [
    [1.0],
    [2.0],
    [3.0],
    [4.0]]
ys = [
    2.0,
    4.0,
    6.0,
    8.0
]


for epoch in range(200):

    total_loss = None

    optimizer.zero_grad()

    for x, y_true in zip(xs, ys):

        pred = model(x)

        loss = (pred - y_true) ** 2

        if total_loss is None:
            total_loss = loss
        else:
            total_loss += loss

    total_loss.backward()

    optimizer.step()

    if epoch % 10 == 0:
        print(epoch, total_loss.data)


