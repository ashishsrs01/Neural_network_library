from tensor import Tensor
from mlp import MLP
from optimizer import SGD

model = MLP(2, [4,1])

optimizer = SGD(model.parameters(), lr = 0.01)

xs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
]

ys = [
    0.0,
    1.0,
    1.0,
    0.0
]


for epoch in range(1000):

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

    if epoch % 100 == 0:
        print(epoch, total_loss.data)


print("\nPredictions:")

for x in xs:
    print(x, "->", model(x))