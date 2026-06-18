# Mini Neural Network Library

A lightweight, educational deep learning library built from scratch using pure Python. This project implements core neural network components including automatic differentiation, computational graphs, and multi-layer perceptrons to provide practical insights into how modern deep learning frameworks like PyTorch and TensorFlow function internally.

**Status:** Production-ready for educational purposes | **License:** Apache 2.0

---

## 🎯 Project Overview

Mini Neural Network is designed for students and developers who want to understand the mathematical foundations and implementation details of neural networks. By implementing everything from first principles, this library demystifies the "black box" nature of modern deep learning frameworks.

### Key Learning Outcomes

- **Automatic Differentiation**: Understand reverse-mode autodiff (backpropagation)
- **Computational Graphs**: Learn how operations are recorded and traced
- **Gradient Computation**: Master the chain rule and gradient flow
- **Neural Network Architecture**: Build multi-layer perceptrons from scratch
- **Training Mechanics**: Implement SGD optimization and loss computation

---

## ✨ Features

### Core Components

- **Tensor Class**: N-dimensional array abstraction with automatic gradient tracking
- **Automatic Differentiation (Autograd)**: Reverse-mode automatic differentiation engine
- **Computational Graph**: Dynamic computational graph construction and traversal
- **Backpropagation**: Complete backward pass implementation with topological sorting
- **Neural Network Layers**: 
  - `Neuron`: Single perceptron with configurable activation functions
  - `Layer`: Stack of neurons for dense transformations
  - `MLP`: Multi-layer perceptron for arbitrary architectures
- **Optimizers**: Stochastic Gradient Descent (SGD) implementation
- **Loss Functions**: Mean Squared Error (MSE) loss computation
- **Activation Functions**: Tanh and linear activations

### Supported Operations

- Element-wise addition (+)
- Element-wise subtraction (-)
- Element-wise multiplication (*)
- Matrix multiplication (@)
- Gradient accumulation and backpropagation

---

## 📦 Installation

### Requirements

- Python 3.7+
- NumPy

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mini-neural-network-library.git
cd mini-neural-network-library
```

2. Install dependencies:
```bash
pip install numpy
```

---

## 🚀 Quick Start

### Basic Tensor Operations

```python
from mini_nn.nn.tensor import Tensor

# Create tensors
a = Tensor([2.0, 3.0])
b = Tensor([4.0, 5.0])

# Perform operations
c = a + b
d = c * 2

# Compute gradients
d.backward()

print(f"Gradients of a: {a.grad}")  # [2. 2.]
```

### Build a Simple Neural Network

```python
from mini_nn.nn.mlp import MLP
from mini_nn.nn.optimizer import SGD
from mini_nn.nn.loss import MSELoss

# Create a 2-layer network: 2 inputs -> 4 hidden -> 1 output
model = MLP(n_in=2, n_outs=[4, 1])

# Setup training components
optimizer = SGD(model.parameters(), lr=0.05)
criterion = MSELoss()

# Training data
xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
ys = [0.0, 1.0, 1.0, 0.0]

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    
    total_loss = None
    for x, y_true in zip(xs, ys):
        pred = model(x)
        loss = criterion(pred, y_true)
        total_loss = loss if total_loss is None else total_loss + loss
    
    total_loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss.data}")
```

---

## 📚 Examples

The repository includes practical examples demonstrating different use cases:

### 1. **XOR Problem** (`examples/xor.py`)
Trains a neural network to learn the XOR function—a classic non-linearly separable problem that demonstrates the necessity of hidden layers.

### 2. **Linear Regression** (`examples/linear_regrassion.py`)
Simple linear regression task showing how neural networks can fit continuous functions.

### 3. **Activation Functions Demo** (`examples/activations_demo.py`)
Explores different activation functions and their effects on network training.

To run examples:
```bash
cd mini_nn/examples
python xor.py
python linear_regrassion.py
python activations_demo.py
```

---

## 📁 Project Structure

```
mini-neural-network-library/
│
├── README.md                          # This file
├── LICENSE                            # Apache 2.0 License
│
└── mini_nn/
    │
    ├── nn/                            # Core neural network modules
    │   ├── tensor.py                  # Tensor class with autograd
    │   ├── neuron.py                  # Single perceptron
    │   ├── layer.py                   # Dense layer (multiple neurons)
    │   ├── mlp.py                     # Multi-layer perceptron
    │   ├── optimizer.py               # SGD optimizer
    │   ├── loss.py                    # Loss functions
    │   ├── train.py                   # Training utilities
    │   └── test.py                    # Unit tests
    │
    └── examples/                      # Practical examples
        ├── xor.py                     # XOR classification
        ├── linear_regrassion.py       # Linear regression
        └── activations_demo.py        # Activation functions
```

---

## 🔧 Architecture

### Tensor

The `Tensor` class is the foundation of the library, providing:
- **Data Storage**: NumPy arrays for efficient computation
- **Gradient Tracking**: Automatic gradient accumulation
- **Operation Recording**: Tracks computational history for backpropagation
- **Backward Pass**: Implements reverse-mode autodiff

### Computational Graph

Operations on tensors create a directed acyclic graph (DAG):
- Each operation is a node in the graph
- Edges represent data dependencies
- The graph is traversed in topological order during backpropagation
- Gradients flow backwards through the graph via chain rule

### Neural Network Layers

```
Neuron → Multiple neurons → Layer → Multiple layers → MLP
```

- **Neuron**: Single unit with weights, bias, and activation
- **Layer**: Collection of neurons performing linear transformation
- **MLP**: Stack of layers forming a complete neural network

---

## 📖 Implementation Details

### Automatic Differentiation

The library implements reverse-mode automatic differentiation (backpropagation):

1. **Forward Pass**: Execute operations, recording the computational graph
2. **Topological Sort**: Order tensors for backward traversal
3. **Backward Pass**: Compute gradients using the chain rule
4. **Gradient Accumulation**: Accumulate gradients for repeated variables

### Backpropagation Algorithm

```
1. Set gradient of output to 1.0
2. Traverse graph in reverse topological order
3. For each node, compute gradients of inputs using chain rule
4. Accumulate gradients in parent tensors
```

---

## 🎓 Learning Resources

This library is ideal for:
- Understanding neural network mathematics and implementation
- Learning automatic differentiation concepts
- Studying backpropagation algorithms
- Building intuition about computational efficiency
- Preparing for deep learning framework development

**Recommended Topics to Study:**
- Calculus: Derivatives, chain rule, partial derivatives
- Linear Algebra: Matrix multiplication, transpose operations
- Graph Theory: Topological sorting, DAGs
- Optimization: Gradient descent, SGD, learning rates

---

## 🚧 Future Enhancements

- [ ] Additional activation functions (ReLU, Sigmoid, Softmax)
- [ ] Batch normalization
- [ ] Convolutional layers
- [ ] Recurrent layers (LSTM, GRU)
- [ ] Advanced optimizers (Adam, RMSprop, Momentum)
- [ ] Regularization techniques (Dropout, L1/L2)
- [ ] GPU acceleration support
- [ ] Comprehensive test suite
- [ ] Performance benchmarks

---

## 🤝 Contributing

Contributions are welcome! This is an educational project, and improvements help everyone learn.

To contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Guidelines:**
- Maintain code clarity for educational purposes
- Include docstrings and comments
- Add examples for new features
- Write unit tests for new functionality

---

## 📝 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This library is designed for **educational purposes**. While fully functional, it prioritizes clarity and understanding over production performance. For production machine learning applications, use established frameworks like:
- [PyTorch](https://pytorch.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [JAX](https://github.com/google/jax)

---

## 👤 Author

Educational Deep Learning Project

---

## 🙏 Acknowledgments

- Inspired by educational implementations in the deep learning community
- Mathematical foundations from calculus and linear algebra
- Design philosophy from modern deep learning frameworks

---

## 📞 Support

For questions or issues:
- Open a GitHub Issue for bug reports
- Start a Discussion for questions
- Check existing documentation and examples

---

**Last Updated:** 2024  
**Status:** Actively maintained for educational purposes
├── main.py
├── README.md
│
└── examples/
```

---

# How Autograd Works

Example:

```python
a = x * y
b = a + y
```

Computational Graph:

```text
x ----\
       * ---> a ---\
y ----/             +
                    ---> b
y ----------------/
```

Calling:

```python
b.backward()
```

traverses the graph in reverse order and computes gradients automatically using the chain rule.

---

# Tensor Example

```python
from tensor import Tensor

x = Tensor(2.0)
y = Tensor(3.0)

z = x * y

print(z)
```

Output:

```python
Tensor(data=6.0, grad=0)
```

---

# Backward Pass Example

```python
from tensor import Tensor

x = Tensor(2.0)
y = Tensor(3.0)

a = x * y
b = a + y

b.backward()

print(x)
print(y)
```

Output:

```python
Tensor(data=2.0, grad=3.0)
Tensor(data=3.0, grad=3.0)
```

---

# Mathematics Behind the Library

Loss Function:

```math
L = (wx - y)^2
```

Gradient calculation:

```math
dL/dw = (dL/da) * (da/dw)
```

The library uses:
- derivatives
- local gradients
- chain rule
- backpropagation

to compute gradients automatically.

---

# Current Implementation

Implemented methods:

```python
__add__()
__mul__()
backward()
```

Internal autograd components:
- `_prev`
- `_op`
- `_backward`

---

# Future Roadmap

## Tensor Operations
- Subtraction
- Division
- Matrix multiplication
- Broadcasting
- Power operator

## Activation Functions
- ReLU
- Sigmoid
- Tanh
- Softmax

## Neural Networks
- Neuron class
- Dense layer
- MLP (Multi-Layer Perceptron)

## Training Features
- Loss functions
- SGD optimizer
- Adam optimizer

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/neural-network-library.git
```

Move into the project directory:

```bash
cd neural-network-library
```

Run the project:

```bash
python main.py
```

---

# Example Computational Graph

```python
x = Tensor(2.0)
y = Tensor(3.0)

a = x * y
b = a + y

b.backward()
```

Backward flow:

```text
b -> a -> x,y
```

Gradients:
- `x.grad = 3`
- `y.grad = 3`

---

# Technologies Used

- Python
- Object-Oriented Programming
- Graph Traversal Algorithms
- Reverse-Mode Automatic Differentiation

---

# Inspiration

Inspired by:
- micrograd
- tinygrad
- modern deep learning frameworks

---

# Author

Ashish Sharma  
AI & Data Science Student

---

# License

This project is open-source and available under the Apache 2 licence.
