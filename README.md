# 🧠 PyDL

> A lightweight Deep Learning library written entirely in pure Python.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![MIT License](https://img.shields.io/badge/License-MIT-green)
![Pure Python](https://img.shields.io/badge/Dependencies-0-orange)

# Understanding Deep Learning

Before using PyDL, it's important to understand how a neural network learns.

## The Artificial Neuron

A neuron receives several inputs, multiplies each one by a **weight**, adds a **bias**, and produces an output.

<math block value="z=(x_1w_1+x_2w_2+\\cdots+x_nw_n)+b"/>

Where:

- **x** → input values
- **w** → learnable weights
- **b** → bias
- **z** → raw output of the neuron

The weights determine the importance of each input, while the bias shifts the result.

## Activation Function

The value **z** is passed through an activation function to create the final output.

<math block value="a=f(z)"/>

Without activation functions, a neural network would only be able to learn linear relationships.

Common examples:

| Activation | Output |
|------------|--------|
| ReLU | `max(0, x)` |
| Sigmoid | Values between 0 and 1 |
| Tanh | Values between -1 and 1 |

## Why Do We Need Derivatives?

A neural network learns by reducing its prediction error.

After making a prediction:

1. Compute the error.
2. Measure how much each weight contributed.
3. Update every weight slightly.

This requires the **derivative** of the activation function.

The derivative tells us **how sensitive the output is to a small change in the input**.

For example, the sigmoid function is:

<math block value="\\sigma(x)=\\frac{1}{1+e^{-x}}"/>

Its derivative is:

<math block value="\\sigma'(x)=\\sigma(x)(1-\\sigma(x))"/>

ReLU is even simpler:

<math block value="\\text{ReLU}(x)=\\max(0,x)"/>

Derivative:

<math block value="\\text{ReLU}'(x)=\\begin{cases}1&x>0\\\\0&x\\le0\\end{cases}"/>

PyDL implements both the **forward function** and its **derivative** for every activation.

## Forward & Backward Propagation

During training, the network repeats the same cycle thousands of times:

```text
Input
  │
  ▼
Weighted Sum (Weights + Bias)
  │
  ▼
Activation Function
  │
  ▼
Prediction
  │
  ▼
Error Calculation
  │
  ▼
Backpropagation
  │
  ▼
Update Weights
```

This iterative optimization process is what allows the network to gradually learn complex patterns from data.

## Why PyDL?

PyDL is an educational deep learning library built from scratch using only Python's standard library. It exposes every part of a neural network—from weight initialization to backpropagation—making it ideal for learning and experimentation.

```python
from pydl import *

nn = Neuronal_network([1, 10, 10, 1], Initialization.he)
```

## Overview

This project is a fully functional implementation of a **feedforward neural network** built entirely with Python's standard library. It is designed for educational purposes, experimentation, and understanding how neural networks work internally without relying on machine learning frameworks.

### Features

* Fully connected neural network architecture
* Forward propagation
* Backpropagation training
* 12 built-in activation functions
* Xavier and He weight initialization
* Custom weight initialization
* Optional Softmax output layer
* Model evaluation and accuracy metrics
* Learning rate finder
* JSON model saving and loading
* Deep model cloning
* Zero external dependencies

---

# Installation

Simply place the library in your project.

```text
project/
│
├── neuronal_network.py
└── main.py
```

Requirements:

* Python 3.10+

No installation or package manager is required.

---

# Quick Start

```python
from neuronal_network import *

# Create a neural network
network = Neuronal_network(
    neuronal_structure=[2, 4, 1],
    init=Initialization.xavier
)

# Set activation function
network.set_activation(
    Activation(ReLU.forward, ReLU.derivative)
)

# XOR dataset
inputs = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]

targets = [
    [0],
    [1],
    [1],
    [0]
]

# Train the model
for _ in range(5000):
    network.train(inputs, targets, learning_rate=0.1)

# Predict
print(network.predict([0,1]))
```

---

# Network Architecture

The architecture is defined by a list of integers.

```python
[3, 8, 8, 2]
```

Represents:

| Layer  | Neurons |
| ------ | ------- |
| Input  | 3       |
| Hidden | 8       |
| Hidden | 8       |
| Output | 2       |

Example:

```python
network = Neuronal_network(
    [4,16,3],
    Initialization.he
)
```

---

# Activation Functions

The library includes multiple activation functions.

| Function   | Description                   |
| ---------- | ----------------------------- |
| Linear     | Identity activation           |
| Sigmoid    | Binary classification         |
| Tanh       | Zero-centered activation      |
| ReLU       | Most common hidden activation |
| LeakyReLU  | Prevents dead neurons         |
| Softplus   | Smooth ReLU                   |
| ELU        | Exponential Linear Unit       |
| GELU       | Transformer activation        |
| Swish      | Google's smooth activation    |
| Mish       | Self-regularized activation   |
| SELU       | Self-normalizing networks     |
| BinaryStep | Threshold activation          |

### Example

```python
network.set_activation(
    Activation(Tanh.forward, Tanh.derivative)
)
```

For activations requiring parameters:

```python
network.set_activation(
    Activation(
        LeakyReLU.forward,
        LeakyReLU.derivative,
        alpha=0.01
    )
)
```

---

# Weight Initialization

Three initialization strategies are available.

## Xavier Initialization

Recommended for Sigmoid and Tanh.

```python
network = Neuronal_network(
    [3,10,1],
    Initialization.xavier
)
```

## He Initialization

Recommended for ReLU-based networks.

```python
network = Neuronal_network(
    [3,10,1],
    Initialization.he
)
```

## Custom Uniform Initialization

```python
initializer = Initialization.uniform(-1,1)

network = Neuronal_network(
    [3,6,2],
    initializer
)
```

---

# Training

Train using gradient descent and backpropagation.

```python
network.train(
    train_list=inputs,
    result_list=targets,
    learning_rate=0.05
)
```

For manual iteration:

```python
for epoch in range(1000):
    network.train(inputs, targets, 0.05)
```

---

# Prediction

```python
result = network.predict([1,0])

print(result)
```

Output:

```python
[0.973]
```

---

# Forward Pass Inspection

Retrieve every layer's output.

```python
layers = network.forward_pass([1,0])

print(layers)
```

Example output:

```python
[
    [1,0],
    [0.34, 0.82, 0.11],
    [0.97]
]
```

Useful for debugging and visualization.

---

# Softmax

Enable Softmax on the output layer.

```python
network.set_softmax(True)
```

Prediction:

```python
print(network.predict(image))
```

Example:

```python
[0.02, 0.91, 0.07]
```

Probabilities always sum to **1.0**.

---

# Model Evaluation

## Average Error

```python
error = network.evaluate(inputs, targets)

print(error)
```

Returns the mean absolute prediction error.

## Accuracy

```python
dataset = list(zip(inputs, targets))

accuracy = network.accuracy(
    dataset,
    tolerance=0.1
)

print(f"{accuracy}%")
```

---

# Learning Rate Finder

Automatically estimate a good learning rate.

```python
best_lr = network.learning_rate_finder(
    min_max=(0.0001, 1),
    training_list=list(zip(inputs, targets)),
    list_question=list(zip(inputs, targets)),
    number_train=8,
    log=True
)

print(best_lr)
```

The algorithm repeatedly narrows the search interval using cloned models.

---

# Saving a Model

Save weights and biases as JSON.

```python
network.save_json("model.json")
```

Example file:

```json
{
    "weights": [
        [[0.2, -0.1], [0.7, 0.4]]
    ],
    "bias": [
        [0.1, -0.3]
    ]
}
```

---

# Loading a Model

```python
network.load_json("model.json")
```

The architecture must match the saved model.

---

# Clone a Network

Create an independent deep copy.

```python
copy = network.clone()

copy.train(inputs, targets, 0.1)
```

The original model remains unchanged.

---

# Reset Parameters

Reinitialize every neuron.

```python
network.reset(Initialization.he)
```

This resets:

* Weights
* Biases
* Internal training state

---

# API Reference

## Neuronal_network

| Method               | Description              |
| -------------------- | ------------------------ |
| `predict()`          | Compute output           |
| `forward_pass()`     | Return every layer       |
| `train()`            | Train on a dataset       |
| `back_propagation()` | Single optimization step |
| `evaluate()`         | Mean absolute error      |
| `accuracy()`         | Percentage accuracy      |
| `clone()`            | Deep copy                |
| `reset()`            | Reinitialize parameters  |
| `save_json()`        | Save model               |
| `load_json()`        | Load model               |
| `set_activation()`   | Change activation        |
| `set_softmax()`      | Enable Softmax           |

---

# Project Structure

```text
neuronal_network.py

├── Activation Functions
│   ├── Linear
│   ├── Sigmoid
│   ├── Tanh
│   ├── ReLU
│   ├── LeakyReLU
│   ├── GELU
│   ├── Swish
│   ├── Mish
│   ├── ELU
│   ├── SELU
│   ├── Softplus
│   └── BinaryStep
│
├── Activation Wrapper
│
├── Initialization
│   ├── Xavier
│   ├── He
│   └── Uniform
│
├── Neuron
│
├── Evaluation
│
├── Model Manager
│
├── Hyperparameter Tuner
│
├── Training
│
└── Neuronal_network
```

---

# Example: Multi-Class Classification

```python
network = Neuronal_network(
    [4,16,3],
    Initialization.he
)

network.set_activation(
    Activation(ReLU.forward, ReLU.derivative)
)

network.set_softmax(True)

prediction = network.predict([5.1, 3.5, 1.4, 0.2])

print(prediction)
```

Output:

```python
[0.97, 0.02, 0.01]
```

---

# Notes

* Written entirely in pure Python.
* Uses standard gradient descent.
* Designed for learning rather than production-scale performance.
* Easily extensible with custom activation functions and initialization strategies.

---

## Contributing

Contributions, ideas, and pull requests are welcome!

If you enjoy PyDL, consider giving the repository a ⭐.

# License

MIT License — free to use, modify, and distribute.
