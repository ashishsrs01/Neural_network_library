# Neural Network Library From Scratch

A lightweight deep learning and automatic differentiation library built completely from scratch using pure Python.

This project is focused on learning how frameworks like PyTorch and TensorFlow work internally by implementing:
- tensors
- gradients
- computational graphs
- backpropagation
- autograd

from scratch.

---

# Features

## Current Features

- Tensor class
- Automatic differentiation (Autograd)
- Computational graph construction
- Gradient storage
- Backward propagation
- Recursive graph traversal
- Topological sorting
- Addition operation
- Multiplication operation
- Gradient accumulation

---

# Learning Goals

This project helps in understanding:
- Calculus for Machine Learning
- Derivatives and Gradients
- Chain Rule
- Backpropagation
- Reverse-mode automatic differentiation
- Neural network internals

---

# Project Structure

```text
neural-network-library/
│
├── tensor.py
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
