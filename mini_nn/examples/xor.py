from mlp import MLP
from optimizer import SGD
from loss import MSELoss

model = MLP(2, [4, 1])

optimizer = SGD(model.parameters(), lr=0.05)

criterion = MSELoss()

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

for epoch in range(5000):

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

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: {total_loss.data}")

print("\nXOR Predictions")

for x in xs:
    print(f"{x} -> {model(x)}")