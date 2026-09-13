# PyDL
Lightweight neural network library written in pure Python.
# PyDL

**PyDL** is a lightweight neural network library built from scratch in pure Python.
Its goal is to provide a simple, readable, and fully customizable implementation of feed-forward neural networks and backpropagation.

## Features

* Dense neural networks
* Backpropagation training
* Multiple activation functions
* Xavier, He and Uniform initialization
* JSON model saving/loading
* Learning rate tuning
* Pure Python (no NumPy required)

## Installation

Clone the repository:

```bash
git clone https://github.com/yourname/PyDL.git
cd PyDL
```

No external dependencies are required.

## Quick Start

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

# Training
X = [[0,0],[0,1],[1,0],[1,1]]
Y = [[0],[1],[1],[0]]

for _ in range(5000):
    nn.train(X, Y, 0.01)

print(nn.predict([1,0]))
```

## Available Activation Functions

| Activation | Output              |
| ---------- | ------------------- |
| Linear     | (-∞, +∞)            |
| ReLU       | [0, +∞)             |
| LeakyReLU  | (-∞, +∞)            |
| Sigmoid    | (0, 1)              |
| Tanh       | (-1, 1)             |
| GELU       | Smooth ReLU         |
| ELU        | Negative saturation |
| Softplus   | Positive smooth     |

## Weight Initialization

* Xavier (Glorot)
* He
* Uniform

## Saving a Model

```python
nn.save_json("model.json")
```

## Loading a Model

```python
nn.load_json("model.json")
```

## Project Structure

```text
PyDL/
│
├── pydl.py
├── examples/
│   ├── xor.py
│   ├── cosine.py
│   └── pong.py
├── README.md
├── LICENSE
└── .gitignore
```

## Roadmap

* [x] Feed-forward neural networks
* [x] Backpropagation
* [x] JSON serialization
* [x] Multiple activation functions
* [ ] Mini-batch training
* [ ] Adam optimizer
* [ ] Cross-entropy loss
* [ ] Convolutional layers (CNN)

## License

This project is released under the MIT License.
