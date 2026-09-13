# PyDL

**PyDL** is a lightweight deep learning library written entirely in pure Python.

Its purpose is to help understand how neural networks work by implementing every important component from scratch: neurons, forward propagation, backpropagation, weight initialization, activation functions and model serialization.

---

# Features

* Fully connected neural networks
* Backpropagation training
* Multiple activation functions
* Xavier, He and Uniform initialization
* JSON save & load
* Learning rate finder
* Pure Python (no NumPy required)

---

# Installation

```bash
git clone https://github.com/Adams54-eng/PyDL.git
cd PyDL
```

No external dependencies are required.

---

# Quick Start

```python
import pydl

nn = pydl.Neuronal_network(
    [2, 8, 8, 1],
    pydl.Initialization.he
)

nn.set_activation(
    pydl.Activation(
        pydl.ReLU.forward,
        pydl.ReLU.derivative
    )
)

nn.neurones[-1][0].activation = pydl.Activation(
    pydl.Tanh.forward,
    pydl.Tanh.derivative
)

X = [[0,0],[0,1],[1,0],[1,1]]
Y = [[0],[1],[1],[0]]

for _ in range(5000):
    nn.train(X, Y, 0.01)

print(nn.predict([1,0]))
```

---

# Creating a Network

A network is defined by its architecture.

```python
nn = pydl.Neuronal_network(
    [4, 16, 8, 2],
    pydl.Initialization.xavier
)
```

The list represents:

* **4** input values
* **16** neurons
* **8** neurons
* **2** output neurons

---

# Activation Functions

Apply one activation to every neuron.

```python
nn.set_activation(
    pydl.Activation(
        pydl.ReLU.forward,
        pydl.ReLU.derivative
    )
)
```

The output layer can use another activation.

```python
nn.neurones[-1][0].activation = pydl.Activation(
    pydl.Linear.forward,
    pydl.Linear.derivative
)
```

Available activations:

| Function  | Output                 |
| --------- | ---------------------- |
| Linear    | (-∞, +∞)               |
| ReLU      | [0, +∞)                |
| LeakyReLU | (-∞, +∞)               |
| Sigmoid   | (0, 1)                 |
| Tanh      | (-1, 1)                |
| ELU       | Smooth negative values |
| GELU      | Smooth ReLU            |
| Softplus  | Positive smooth output |

---

# API Reference

## Neuronal_network

### `__init__(neuronal_structure, init)`

Creates a neural network.

```python
nn = pydl.Neuronal_network(
    [3, 12, 1],
    pydl.Initialization.he
)
```

| Parameter            | Description                  |
| -------------------- | ---------------------------- |
| `neuronal_structure` | List describing each layer   |
| `init`               | Weight initialization method |

---

### `set_activation(activation)`

Applies the same activation to every neuron.

```python
nn.set_activation(
    pydl.Activation(
        pydl.ReLU.forward,
        pydl.ReLU.derivative
    )
)
```

---

## Prediction

### `predict(input)`

Returns the network prediction.

```python
result = nn.predict([0.5, 0.1])
```

**Returns**

```python
[0.842]
```

---

### `forward_pass(input)`

Returns the output of every layer.

```python
layers = nn.forward_pass([1,0])
```

Example output:

```python
[
    [1,0],
    [...],
    [...],
    [0.91]
]
```

Useful for debugging or visualizing hidden layers.

---

## Training

### `train(train_list, result_list, learning_rate)`

Trains the network for one complete epoch.

```python
nn.train(X, Y, 0.001)
```

| Parameter       | Description           |
| --------------- | --------------------- |
| `train_list`    | Input samples         |
| `result_list`   | Expected outputs      |
| `learning_rate` | Gradient descent step |

---

### `back_propagation(input, expected, learning_rate)`

Performs a single learning step.

```python
nn.back_propagation(
    [1,0],
    [1],
    0.001
)
```

This method automatically performs:

1. Forward propagation
2. Error computation
3. Gradient propagation
4. Weight update

---

## Evaluation

### `evaluate(inputs, expected)`

Computes the average absolute error.

```python
error = nn.evaluate(X, Y)
```

Example:

```python
0.0134
```

---

### `accuracy(dataset, tolerance)`

Computes prediction accuracy within a tolerance.

```python
acc = nn.accuracy(dataset, 0.05)
```

Example:

```python
98.7
```

---

## Model Management

### `save_json(path)`

Saves every weight and bias.

```python
nn.save_json("model.json")
```

The file contains:

```json
{
    "weights": [...],
    "bias": [...]
}
```

---

### `load_json(path)`

Loads a previously trained model.

```python
nn.load_json("model.json")
```

---

### `clone()`

Creates a deep copy of the network.

```python
copy = nn.clone()
```

Useful for testing hyperparameters without modifying the original model.

---

### `reset()`

Reinitializes every weight using the original initialization method.

```python
nn.reset()
```

---

## Hyperparameter Tuning

### `learning_rate_finder(min_max, training_list, validation_list, number_train, log=False)`

Automatically searches for a good learning rate.

```python
lr = nn.learning_rate_finder(
    (1e-5, 0.1),
    train_data,
    valid_data,
    6
)
```

Returns the best learning rate found.

---

# Weight Initialization

## He Initialization

Recommended with ReLU.

```python
pydl.Initialization.he
```

## Xavier Initialization

Balanced initialization for many activation functions.

```python
pydl.Initialization.xavier
```

## Uniform Initialization

Custom interval.

```python
pydl.Initialization.uniform(-1, 1)
```

---

# Examples

Approximate a cosine function:

```python
import math

for angle in range(-360, 361):
    x = [angle / 360]
    y = [math.cos(math.radians(angle))]
    nn.back_propagation(x, y, 0.001)
```

Predict:

```python
print(nn.predict([45/360]))
```

---

# Project Structure

```text
PyDL/
│
├── pydl.py
├── README.md
├── LICENSE
├── examples/
│   ├── xor.py
│   ├── cosine.py
│   ├── tangent.py
│   └── visualizer.py
└── models/
    └── model.json
```

---

# Philosophy

PyDL is **not designed to compete with TensorFlow or PyTorch**. It is an educational library whose objective is to make neural networks understandable, hackable and easy to extend.

Every algorithm is intentionally kept readable so anyone can modify the source code and experiment with artificial intelligence from scratch.
