import math
import random
import pydl
from pydl import *

# ==========================================================
# Neural Network Example
# Learn the cosine function: y = cos(x)
# ==========================================================

# Create a neural network with:
# 1 input → 3 hidden layers of 10 neurons → 1 output
nn = pydl.Neuronal_network(
    [1, 10, 10, 10, 1],
    pydl.Initialization.he
)

# Use ReLU for every hidden layer
nn.set_activation(
    pydl.Activation(
        pydl.ReLU.forward,
        pydl.ReLU.derivative
    )
)

# Replace the output activation with Tanh
# since cos(x) is naturally bounded between -1 and 1.
nn.neurones[-1][0].activation = pydl.Activation(
    pydl.Tanh.forward,
    pydl.Tanh.derivative
)

# ==========================================================
# Training
# ==========================================================

for _ in range(100000):

    # Generate a random angle in degrees
    angle = random.randint(-360, 360)

    # Expected output: cosine of the angle
    target = [math.cos(math.radians(angle))]

    # Normalize the input to the range [-1, 1]
    x = [angle / 360]

    # Perform one backpropagation step
    nn.back_propagation(x, target, 0.001)

# ==========================================================
# Save the trained model
# ==========================================================

nn.save_json("data.json")

print("Training complete! Model saved to data.json")
