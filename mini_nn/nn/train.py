from mlp import MLP
from optimizer import SGD

model = MLP(1, [1])

optimizer = SGD(model.parameters(), lr = 0.01)

x = [2.0]
target = 10.0

for epoch in range(100):

    pred = model(x)
    loss = (pred - target)**2

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch}: Loss = {loss.data}")
