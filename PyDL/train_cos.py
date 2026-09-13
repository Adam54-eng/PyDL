import pydl
import math
import random
from pydl import *

nn = pydl.Neuronal_network([1, 10, 10, 10, 1], pydl.Initialization.he)
nn.set_activation(pydl.Activation(pydl.ReLU.forward, pydl.ReLU.derivative))
nn.neurones[-1][0].activation = pydl.Activation(pydl.Tanh.forward, pydl.Tanh.derivative)

for _ in range(100000):

    angle = random.randint(-360, 360)
    result = [math.cos(math.radians(angle))]
    angle = [angle / 360]
    nn.back_propagation(angle, result, 0.001)

nn.save_json("data.json")
