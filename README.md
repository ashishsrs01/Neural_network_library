<p align="center">
  <h1 align="center">⚡ Mini Neural Network Library</h1>
  <p align="center">
    A lightweight deep learning library built from scratch in pure Python.<br/>
    Learn how autograd, backpropagation, and neural networks really work—by reading the code.
  </p>
</p>

<p align="center">
  <a href="https://ashishsrs01.github.io/Neural_network_library/">Live Demo</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-examples">Examples</a> •
  <a href="#-contributing">Contributing</a>
</p>

<p align="center">
  <a href="https://ashishsrs01.github.io/Neural_network_library/"><img src="https://img.shields.io/badge/📄_docs-GitHub%20Pages-blue?style=flat" alt="GitHub Pages"/></a>
  <img src="https://img.shields.io/badge/python-3.7+-blue?logo=python&logoColor=white" alt="Python 3.7+"/>
  <img src="https://img.shields.io/badge/numpy-required-013243?logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"/>
  <img src="https://img.shields.io/badge/purpose-educational-orange" alt="Educational"/>
</p>

---

## 🎯 Why This Project?

Modern deep learning frameworks like PyTorch and TensorFlow are powerful—but they hide the mechanics behind layers of abstraction. **Mini Neural Network** strips everything back to first principles so you can see exactly how a neural network learns.

By studying (and modifying) this library, you will understand:

| Concept | What You'll Learn |
|---|---|
| **Automatic Differentiation** | How reverse-mode autodiff (backpropagation) computes gradients without manual calculus |
| **Computational Graphs** | How operations are recorded as a DAG and traversed during the backward pass |
| **Gradient Flow** | How the chain rule propagates gradients through every operation |
| **Network Architecture** | How neurons, layers, and MLPs compose into trainable models |
| **Training Loop** | How SGD, loss functions, and gradient zeroing work together |

---

## ✨ Features

### Core Engine

| Component | Description |
|---|---|
| **Tensor** | N-dimensional array with automatic gradient tracking (backed by NumPy) |
| **Autograd** | Full reverse-mode automatic differentiation engine |
| **Computational Graph** | Dynamic DAG construction with topological-sort backward pass |

### Neural Network Modules

| Module | Description |
|---|---|
| `Neuron` | Single perceptron with configurable activation |
| `Layer` | Dense layer (stack of neurons) |
| `MLP` | Multi-layer perceptron supporting arbitrary architectures |
| `SGD` | Stochastic Gradient Descent optimizer |
| `MSELoss` | Mean Squared Error loss function |

### Supported Operations

| Category | Operations |
|---|---|
| **Arithmetic** | `+`  `−`  `*`  `/`  `**` (power)  negation |
| **Linear Algebra** | `@` matrix multiplication |
| **Activations** | `tanh`  `relu`  `sigmoid` |
| **Reductions** | `sum`  `mean` |
| **Autograd** | `backward()` with full gradient accumulation |

> Every operation listed above supports automatic gradient computation.

---

## 📦 Installation

### Prerequisites

- Python 3.7+
- NumPy

### Setup

```bash
# Clone the repository
git clone https://github.com/ashishsrs01/Neural_network_library.git
cd Neural_network_library

# Install the only dependency
pip install numpy
```

---

## 🚀 Quick Start

### 1. Tensor Operations & Autograd

```python
from mini_nn.nn.tensor import Tensor

# Create tensors
a = Tensor([2.0, 3.0])
b = Tensor([4.0, 5.0])

# Forward pass — operations build the computational graph
c = a + b        # element-wise addition
d = c * 2        # scalar multiplication

# Backward pass — gradients flow automatically
d.backward()

print(a.grad)    # [2. 2.]
print(b.grad)    # [2. 2.]
```

### 2. Training an MLP on XOR

```python
from mini_nn.nn.mlp import MLP
from mini_nn.nn.optimizer import SGD
from mini_nn.nn.loss import MSELoss

# Architecture: 2 inputs → 4 hidden (tanh) → 1 output (linear)
model = MLP(n_in=2, n_outs=[4, 1])
optimizer = SGD(model.parameters(), lr=0.05)
criterion = MSELoss()

# XOR dataset
xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
ys = [0.0, 1.0, 1.0, 0.0]

# Training loop
for epoch in range(5000):
    optimizer.zero_grad()

    total_loss = None
    for x, y_true in zip(xs, ys):
        pred = model(x)
        loss = criterion(pred, y_true)
        total_loss = loss if total_loss is None else total_loss + loss

    total_loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss.data:.6f}")

# Evaluate
for x in xs:
    print(f"{x} → {model(x).data:.4f}")
```

---

## 📁 Project Structure

```
Neural_network_library/
├── README.md
├── LICENSE                          # Apache 2.0
├── index.html                       # Interactive documentation page
│
└── mini_nn/
    ├── nn/                          # Core library
    │   ├── tensor.py                # Tensor class with full autograd
    │   ├── neuron.py                # Single perceptron unit
    │   ├── layer.py                 # Dense layer
    │   ├── mlp.py                   # Multi-layer perceptron
    │   ├── optimizer.py             # SGD optimizer
    │   ├── loss.py                  # MSE loss function
    │   ├── train.py                 # Standalone training script (XOR)
    │   └── test.py                  # Unit tests
    │
    └── examples/                    # Ready-to-run demos
        ├── xor.py                   # XOR classification
        ├── linear_regrassion.py     # Linear regression
        └── activations_demo.py      # Activation function exploration
```

---

## 🔧 Architecture

### Computational Graph

Every operation on `Tensor` objects records a node in a directed acyclic graph (DAG). When `backward()` is called, the graph is traversed in **reverse topological order** and gradients are propagated via the chain rule.

```
Forward Pass                          Backward Pass

x ────┐                               x.grad ← ∂b/∂x
      mul → a ──┐                     a.grad ← ∂b/∂a
y ────┘         add → b               b.grad ← 1
      ──────────┘
y ──────────────┘                     y.grad ← ∂b/∂y
```

### Layer Hierarchy

```
Tensor
  └── Neuron   (weights · inputs + bias → activation)
       └── Layer    (n parallel neurons)
            └── MLP      (sequential stack of layers)
```

- **Hidden layers** apply `tanh` activation by default.
- The **output layer** is linear (no activation), making the MLP suitable for regression out of the box.

### Backpropagation Algorithm

```
1.  Set output gradient to 1.0
2.  Topologically sort the computational graph
3.  For each node (in reverse order):
      a. Compute local gradients with respect to each input
      b. Multiply by the incoming gradient (chain rule)
      c. Accumulate into the input tensors' .grad fields
4.  Optimizer reads .grad and updates .data
```

---

## 📚 Examples

### XOR Classification — `examples/xor.py`

Trains a 2→4→1 MLP on the XOR function, a classic non-linearly separable problem that demonstrates why hidden layers are necessary.

### Linear Regression — `examples/linear_regrassion.py`

Fits a simple linear function to show how neural networks approximate continuous mappings.

### Activation Functions — `examples/activations_demo.py`

Explores different activation functions and their effect on network training.

**Run any example:**

```bash
cd mini_nn/examples
python xor.py
python linear_regrassion.py
python activations_demo.py
```

---

## 📖 Implementation Details

### How Autograd Works — A Concrete Example

```python
from mini_nn.nn.tensor import Tensor

x = Tensor(2.0)
y = Tensor(3.0)

a = x * y           # a = 6.0
b = a + y           # b = 9.0

b.backward()

print(x.grad)       # 3.0  →  ∂b/∂x = y
print(y.grad)       # 3.0  →  ∂b/∂y = x + 1
```

**Why `y.grad = 3.0`?** Variable `y` participates in *two* operations (`a = x * y` and `b = a + y`), so its gradient is the *sum* of both paths through the graph:

```
∂b/∂y = (∂b/∂a)(∂a/∂y) + (∂b/∂y_direct) = (1)(x) + (1) = 2 + 1 = 3
```

### Mathematics of the Training Loop

The library minimizes a loss function via gradient descent:

$$L = \sum_i (f(x_i) - y_i)^2$$

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w}$$

$$w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}$$

The autograd engine computes $\frac{\partial L}{\partial w}$ automatically—you never write derivative code by hand.

---

## 🎓 Recommended Background

To get the most out of this library, familiarity with these topics is helpful:

- **Calculus** — Derivatives, chain rule, partial derivatives
- **Linear Algebra** — Dot products, matrix multiplication
- **Graph Theory** — DAGs, topological sorting
- **Optimization** — Gradient descent, learning rates

---

## 🚧 Roadmap

- [ ] Softmax activation & cross-entropy loss
- [ ] Batch normalization
- [ ] Convolutional layers
- [ ] Recurrent layers (LSTM, GRU)
- [ ] Advanced optimizers (Adam, RMSprop, Momentum)
- [ ] Regularization (Dropout, L1/L2)
- [ ] GPU acceleration
- [ ] Comprehensive test suite & benchmarks

---

## 🤝 Contributing

Contributions are welcome! This is an educational project—improvements help everyone learn.

```bash
# 1. Fork the repository
# 2. Create a feature branch
git checkout -b feature/your-feature

# 3. Make your changes, then commit
git commit -m "Add your feature"

# 4. Push and open a Pull Request
git push origin feature/your-feature
```

**Guidelines:**
- Prioritise code clarity—this is a teaching tool
- Include docstrings and inline comments
- Add examples for new features
- Write unit tests for new functionality

---

## 📝 License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

---

## ⚠️ Disclaimer

This library is designed for **educational purposes**. It is fully functional but prioritises clarity over production performance. For production workloads, use established frameworks:

- [PyTorch](https://pytorch.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [JAX](https://github.com/google/jax)

---

## 👤 Author

**Ashish Sharma** — AI & Data Science Student

---

## 🙏 Acknowledgments

- Inspired by [micrograd](https://github.com/karpathy/micrograd) and [tinygrad](https://github.com/tinygrad/tinygrad)
- Mathematical foundations from calculus and linear algebra
- Design philosophy drawn from modern deep learning frameworks

---

<p align="center">
  <sub>Built for learning. Read the source. Break things. Rebuild them better.</sub>
</p>
