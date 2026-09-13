# PyDL

**PyDL** is a lightweight deep learning library written entirely in pure Python.
It was created to understand **how neural networks work internally**, without relying on NumPy, TensorFlow or PyTorch.

Instead of hiding the mathematics, PyDL exposes every step: neurons, weights, activations, forward propagation and backpropagation.

---

# Why PyDL?

Most AI libraries are optimized for performance. PyDL is optimized for **learning**.

With only a few lines of code, you can create, train, save and reuse fully connected neural networks while keeping the implementation easy to read and modify.

---

# Features

* Fully connected neural networks
* Backpropagation training
* Multiple activation functions
* Xavier, He and Uniform initialization
* JSON model saving and loading
* Learning rate utilities
* Pure Python implementation

---

# How the library works

A neural network in PyDL is composed of **layers of neurons**.

Each neuron contains:

* A list of weights
* One bias
* One activation function

The data flows through the network during **forward propagation**, and the weights are updated during **backpropagation**.

<svg viewBox="0 0 320 140">
  <rect width=320 height=140 rx=12 fill="#F8FAFC" stroke="#CBD5E1"/>
  <g fill="#2563EB">
    <circle cx=34 cy=50 r=8/>
    <circle cx=34 cy=90 r=8/>
  </g>
  <g fill="#64748B">
    <circle cx=112 cy=34 r=7/>
    <circle cx=112 cy=70 r=7/>
    <circle cx=112 cy=106 r=7/>
  </g>
  <g fill="#64748B">
    <circle cx=192 cy=34 r=7/>
    <circle cx=192 cy=70 r=7/>
    <circle cx=192 cy=106 r=7/>
  </g>
  <g fill="#059669">
    <circle cx=286 cy=70 r=8/>
  </g>
  {#each [34,70,106] as y}
    <line x1=42 y1=50 x2=105 y2={y} stroke="#CBD5E1"/>
    <line x1=42 y1=90 x2=105 y2={y} stroke="#CBD5E1"/>
    <line x1=119 y1={y} x2=185 y2=34 stroke="#D1D5DB"/>
    <line x1=119 y1={y} x2=185 y2=70 stroke="#D1D5DB"/>
    <line x1=119 y1={y} x2=185 y2=106 stroke="#D1D5DB"/>
  {/each}
  {#each [34,70,106] as y}
    <line x1=199 y1={y} x2=278 y2=70 stroke="#D1D5DB"/>
  {/each}
  <text x=34 y=18 fontSize=8 textAnchor="middle" fill="#1F2937">Input</text>
  <text x=112 y=18 fontSize=8 textAnchor="middle" fill="#1F2937">Hidden</text>
  <text x=192 y=18 fontSize=8 textAnchor="middle" fill="#1F2937">Hidden</text>
  <text x=286 y=18 fontSize=8 textAnchor="middle" fill="#1F2937">Output</text>
</svg>

---

# Creating a network

The architecture is defined with a list of integers.

```python
import pydl

nn = pydl.Neuronal_network(
    [2, 8, 8, 1],
    pydl.Initialization.he
)
```

This creates:

* 2 input values
* 2 hidden layers with 8 neurons
* 1 output neuron

---

# Activation functions

Choose the activation used by every hidden layer.

```python
nn.set_activation(
    pydl.Activation(
        pydl.ReLU.forward,
        pydl.ReLU.derivative
    )
)
```

Available activations include:

| Function  | Typical use              |
| --------- | ------------------------ |
| Linear    | Regression output        |
| ReLU      | Hidden layers            |
| LeakyReLU | Hidden layers            |
| Sigmoid   | Binary classification    |
| Tanh      | Values between -1 and 1  |
| GELU      | Smooth hidden activation |
| ELU       | Negative saturation      |
| Softplus  | Positive outputs         |

The output layer can use a different activation:

```python
nn.neurones[-1][0].activation = pydl.Activation(
    pydl.Tanh.forward,
    pydl.Tanh.derivative
)
```

---

# Training

Training is performed with backpropagation.

```python
X = [[0,0], [0,1], [1,0], [1,1]]
Y = [[0], [1], [1], [0]]

for _ in range(5000):
    nn.train(X, Y, 0.01)
```

You can also train one sample manually:

```python
nn.back_propagation([1,0], [1], 0.01)
```

---

# Prediction

Once trained:

```python
prediction = nn.predict([1,0])
print(prediction)
```

The network always returns a list of output values.

---

# Saving and loading

Save the learned weights and biases:

```python
nn.save_json("model.json")
```

Reload them later:

```python
nn.load_json("model.json")
```

This makes it possible to train a model once and reuse it instantly.

---

# Initialization methods

PyDL includes three initialization strategies:

| Method  | Description             |
| ------- | ----------------------- |
| Xavier  | Balanced initialization |
| He      | Recommended for ReLU    |
| Uniform | Custom random interval  |

Example:

```python
nn = pydl.Neuronal_network(
    [1,16,16,1],
    pydl.Initialization.xavier
)
```

---

# Example projects

The library can already be used for projects such as:

* XOR solver
* Cosine approximation
* Tangent approximation
* Function visualizer with Tkinter
* Small AI experiments

---

# Philosophy

PyDL is designed for **education**, **experimentation**, and **understanding neural networks from scratch**.

Every algorithm is intentionally kept readable so that you can modify the source code and build your own AI systems.
