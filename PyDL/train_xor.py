import random
import pydl
from pydl import *

# ==========================================================
# Neural Network Example
# Learn the XOR logical function
# ==========================================================

# Create a neural network with:
# 2 inputs → 3 hidden layers of 10 neurons → 1 output
nn = pydl.Neuronal_network(
    [2, 10, 10, 10, 1],
    pydl.Initialization.he
)

# Use ReLU for every hidden layer
nn.set_activation(
    pydl.Activation(
        pydl.ReLU.forward,
        pydl.ReLU.derivative
    )
)

# Replace the output activation with Sigmoid
# since XOR outputs values between 0 and 1.
nn.neurones[-1][0].activation = pydl.Activation(
    pydl.Sigmoid.forward,
    pydl.Sigmoid.derivative
)

# ==========================================================
# Training
# ==========================================================

xor_data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
]

for _ in range(100000):

    # Select a random XOR example
    x, target = random.choice(xor_data)

    # Perform one backpropagation step
    nn.back_propagation(x, target, 0.001)

# ==========================================================
# Save the trained model
# ==========================================================

nn.save_json("xor_model.json")

print("Training complete! Model saved to xor_model.json")

# ==========================================================
# Model Evaluation
# ==========================================================

test_inputs = [x for x, _ in xor_data]
test_targets = [y for _, y in xor_data]

error = nn.evaluate(test_inputs, test_targets)

print(f"Mean Absolute Error: {error:.6f}")

# ==========================================================
# Predictions
# ==========================================================

print("\nXOR Predictions:")

for x, expected in xor_data:
    prediction = nn.predict(x)[0]
    print(f"{x} -> {prediction:.3f} (expected {expected[0]})")
