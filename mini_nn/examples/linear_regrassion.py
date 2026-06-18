from mlp import MLP
from optimizer import SGD
from loss import MSELoss

model = MLP(1, [1])

optimizer = SGD(model.parameters(), lr=0.01)

criterion = MSELoss()

xs = [
    [1.0],
    [2.0],
    [3.0],
    [4.0]
]

ys = [
    2.0,
    4.0,
    6.0,
    8.0
]

for epoch in range(1000):

    optimizer.zero_grad()

    total_loss = None

    for x, y_true in zip(xs, ys):

        pred = model(x)

        loss = criterion(pred, y_true)

        if total_loss is None:
            total_loss = loss
        else:
            total_loss = total_loss + loss

    total_loss.backward()

    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: {total_loss.data}")

print("\nPredictions")

for x in [1.0, 2.0, 5.0]:
    print(f"x={x} -> {model([x])}")