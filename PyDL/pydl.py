import math
import random
import copy
import json

class Linear:

    @staticmethod
    def forward(x):

        return x

    @staticmethod
    def derivative(a):

        return 1

class Tanh:

    @staticmethod
    def forward(x):

        return math.tanh(x)

    @staticmethod
    def derivative(a):

        return 1 - a ** 2

class Sigmoid:

    @staticmethod
    def forward(x):

        return 1 / (1 + math.exp(-x))

    @staticmethod
    def derivative(a):

        return a * (1 - a)

class ReLU:

    @staticmethod
    def forward(x):

        return max(0, x)

    @staticmethod
    def derivative(a):

        return 1 if a > 0 else 0

class LeakyReLU:

    @staticmethod
    def forward(x, alpha=0.01):

        return x if x > 0 else alpha * x

    @staticmethod
    def derivative(a, alpha=0.01):

        return 1 if a > 0 else alpha

class Softplus:

    @staticmethod
    def forward(x):

        return math.log(1 + math.exp(x))

    @staticmethod
    def derivative(x):

        return 1 / (1 + math.exp(-x))

class ELU:

    @staticmethod
    def forward(x, alpha=1.0):

        return x if x > 0 else alpha * (math.exp(x) - 1)

    @staticmethod
    def derivative(a, alpha=1.0):

        return 1 if a > 0 else a + alpha

class GELU:

    @staticmethod
    def forward(x):

        return 0.5 * x * (
            1 + math.tanh(
                math.sqrt(2 / math.pi) *
                (x + 0.044715 * x ** 3)
            )
        )

    @staticmethod
    def derivative(x):

        t = math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)
        tanh_t = math.tanh(t)

        return (
            0.5 * (1 + tanh_t)
            + 0.5 * x
            * (1 - tanh_t ** 2)
            * math.sqrt(2 / math.pi)
            * (1 + 3 * 0.044715 * x ** 2)
        )

class Swish:

    @staticmethod
    def forward(x):

        s = 1 / (1 + math.exp(-x))
        return x * s

    @staticmethod
    def derivative(x):

        s = 1 / (1 + math.exp(-x))
        return s + x * s * (1 - s)

class Mish:

    @staticmethod
    def forward(x):

        return x * math.tanh(math.log1p(math.exp(x)))

    @staticmethod
    def derivative(x):

        sp = math.log1p(math.exp(x))
        tsp = math.tanh(sp)
        sig = 1 / (1 + math.exp(-x))
        return tsp + x * sig * (1 - tsp**2)

class BinaryStep:

    @staticmethod
    def forward(x):

        return 1 if x >= 0 else 0

    @staticmethod
    def derivative(a):

        return 0

class BentIdentity:

    @staticmethod
    def forward(x):

        return ((math.sqrt(x*x + 1) - 1) / 2) + x

    @staticmethod
    def derivative(x):

        return x / (2 * math.sqrt(x*x + 1)) + 1

class SELU:

    @staticmethod
    def forward(x, alpha=1.67326324):

        scale = 1.05070098

        if x > 0:

            return scale * x
        
        return scale * alpha * (math.exp(x) - 1)

    @staticmethod
    def derivative(a, alpha=1.67326324):

        scale = 1.05070098

        if a > 0:

            return scale
        
        return scale * (a + alpha)

class Activation:

    def __init__(self, forward, derivative, alpha=None):

        self.forward = forward
        self.derivative = derivative
        self.alpha = alpha

    @staticmethod
    def numerical_derivative(func, x, h=0.0001):

        x1 = x - (h / 2)
        x2 = x + (h / 2)
        y1 = func(x1)
        y2 = func(x2)

        return (y2 - y1) / (x2 - x1)

class Initialization:

    @staticmethod
    def uniform(a, b):
        """Returns an initializer generating values uniformly between a and b."""

        return lambda n_in, n_out: random.uniform(a, b)

    @staticmethod
    def xavier(n_in, n_out):
        """Generates one weight using Xavier (Glorot) initialization."""

        limit = math.sqrt(6 / (n_in + n_out))
        return random.uniform(-limit, limit)

    @staticmethod
    def he(n_in, n_out):
        """Generates one weight using He initialization."""

        std = math.sqrt(2 / n_in)
        return random.gauss(0, std)

class neurones:

    def __init__(self, initialization: Initialization, n_in: float, n_out: float, activation: Activation):
        """Creates a neuron with initialized weights, zero bias, and an activation function."""

        self.weight = [initialization(n_in, n_out) for _ in range(n_in)]
        self.bias = 0
        self.activation = activation
        self.alpha = activation.alpha

    def retroprogager(self, entree: list[float]) -> float:
        """Computes the neuron's output from the given inputs."""

        e = 0
        for a, b in zip(entree, self.weight):

            e += a * b

        if self.alpha is not None:

            return self.activation.forward(e + self.bias, self.alpha)

        else:

            return self.activation.forward(e + self.bias)

    def back_propagation(self: neurones, entree: list[float], error: float, learning_rate: float):
        """Updates the neuron's weights and bias using backpropagation."""

        self.bias -= error * learning_rate
        self.weight = [w - (error * (learning_rate * e)) for w, e in zip(self.weight, entree)]

    def to_dict(self):
        """Returns a JSON-serializable representation of the neuron."""

        return {
            "weight": self.weight,
            "bias": self.bias
        }

class evaluation:

    def forward_pass(self: Neuronal_network, entree: list[float]) -> list[list[float]]:
        """This method allows for the observation of data transmission between neurons based on input results."""

        e = [entree]
        for x in range(len(self.neurones)):

            couche = []
            for y in self.neurones[x]:
                
                couche.append(y.retroprogager(e[-1]))

            e.append(couche)

        if self.softmax:

            m = max(e[-1])
            exp = [math.exp(x - m) for x in e[-1]]
            s = sum(exp)
            e[-1] = [a / s for a in exp]

        return e

    def predict(self: Neuronal_network, entree: list[float]) -> list[float]:
        """This method engages the neural network."""

        return self.forward_pass(entree)[len(self.neuronal_structure) - 1]

    def evaluate(self: Neuronal_network, list_question: list[list], list_exepted_result: list[list]) -> float:
        """Computes the average absolute error of the network's predictions on a list of question/expected-result pairs."""

        error = []
        for r, r2 in zip(list_question, list_exepted_result):

            result = self.predict(r)
            error.append(abs(sum([abs(p - e) for p, e in zip(result, r2)]) / len(result)))
       
        return sum(error) / len(error)

    def accuracy(self: Neuronal_network, list_question: list[tuple[list, list]], tolerance: float):
        """Calculates the prediction accuracy as a percentage of outputs within the specified tolerance."""

        Error = 0
        for input, s in list_question:

            predict = self.predict(input)
            error = sum(abs(p - e) for p, e in zip(predict, s)) / len(predict)
            if error <= tolerance:

                Error += 1

        return Error / (len(list_question)) * 100

class model_manager:

    def save_json(self: Neuronal_network, file_json):
        """Saves the network's weights and biases to a JSON file."""

        with open(file_json, "w") as f:

            json.dump({
                "weights": [
                    [n.weight for n in couche]
                    for couche in self.neurones
                ],
                "bias": [
                    [n.bias for n in couche]
                    for couche in self.neurones
                ]
            }, f, indent=4)
   
    def load_json(self: Neuronal_network, file_json):
        """Loads the network's weights and biases from a JSON file."""

        with open(file_json, "r") as f:

            data = json.load(f)

        for i, couche in enumerate(self.neurones):

            for j, neurone in enumerate(couche):

                neurone.weight = data["weights"][i][j]
                neurone.bias = data["bias"][i][j]

    def clone(self):
        """Returns a deep copy of the neural network."""

        n = copy.deepcopy(self)
        return n

    def reset(self: Neuronal_network, init: Initialization):
        """Reinitializes all neurons with new random weights and zero biases."""

        self.neurones = []
        self.learning_rate = 0.001
        self.old_error = 0

        for x in range(len(self.neuronal_structure) - 1):

            n_in = self.neuronal_structure[x]
            n_out = self.neuronal_structure[x + 1]
            self.neurones.append([neurones(init, n_in, n_out, Activation(Linear.forward, Linear.derivative)) for _ in range(n_out)])

class hyperparameter_tuner:

    def learning_rate_finder(self: Neuronal_network, min_max: tuple[float, float],  training_list: list[tuple[list, list]], list_question: list[tuple[list, list]], number_train: int, log=False) -> float:
        """Searches for the best learning rate by narrowing an interval, testing candidates on independent copies of the network."""
       
        lr_list = [min_max[0], sum(min_max) / 2, min_max[1]]
        for _ in range(number_train):

            result = []
            for y in lr_list:

                cerveau = self.clone()
                for r, r2 in training_list:

                    cerveau.back_propagation(r, r2, y)
       
                diff = []
                for z, z2 in list_question:

                    prediction = cerveau.predict(z)

                    diff.append(
                        sum(abs(p - e) for p, e in zip(prediction, z2))
                        / len(prediction)
                    )
               
                result.append(sum(diff) / len(diff))

            paires = sorted(zip(lr_list, result), key=lambda p: p[1])
            lr_list = []
            for taux, erreur in paires:

                if log: print(f"learning_rate = {taux}, error = {erreur}")
                lr_list.append(taux)

            if log: print("\n")
            p = paires[:2]
            lr_list = [p[0][0], math.sqrt(p[0][0] * p[1][0]), p[1][0]]
   
        return lr_list[0]
    
class train:

    def back_propagation(self: Neuronal_network, entree: list[float], expected_result: list[float], learning_rate: float):
        """Performs one forward pass and one backpropagation step to update the weights and biases."""

        forward_pass = self.forward_pass(entree)
        Error = []

        for f, e, n in zip(forward_pass[-1], expected_result, self.neurones[-1]):
            
            if n.alpha is None:

                Error.append((f - e) * n.activation.derivative(f))
            else:

                Error.append((f - e) * n.activation.derivative(f, n.alpha))

        error = [Error]
        for x in reversed(range(1, len(self.neurones))):

            couche = [0.0] * len(self.neurones[x - 1])
            for y in range(len(self.neurones[x])):

                for w in range(len(self.neurones[x][y].weight)):

                    couche[w] += Error[y] * self.neurones[x][y].weight[w]

            for i in range(len(couche)):

                neurone = self.neurones[x - 1][i]

                if neurone.alpha is None:

                    couche[i] *= neurone.activation.derivative(forward_pass[x][i])

                else:

                    couche[i] *= neurone.activation.derivative(
                        forward_pass[x][i],
                        neurone.alpha
                    )

            error.insert(0, couche)
            Error = couche

        for x in range(len(self.neurones)):

            for y in range(len(self.neurones[x])):

                self.neurones[x][y].back_propagation(forward_pass[x], error[x][y], learning_rate)
   
    def train(self: Neuronal_network, train_list: list[list], result_list: list[list], learning_rate):
        """Trains the network on a full pass through the training list using backpropagation."""

        for r, r2 in zip(train_list, result_list):

            self.back_propagation(r, r2, learning_rate) 

class Neuronal_network(
    evaluation,
    model_manager,
    hyperparameter_tuner,
    train
):

    def __init__(self: Neuronal_network, neuronal_structure: list[int], init: Initialization):
        """Creates a neural network with random weights and biases based on the given layer structure."""

        #initial variable
        self.neuronal_structure = neuronal_structure
        self.neurones = []
        self.learning_rate = 0.001
        self.old_error = 0
        self.softmax = False

        for x in range(len(self.neuronal_structure) - 1):

            n_in = self.neuronal_structure[x]
            n_out = self.neuronal_structure[x + 1]
            self.neurones.append([neurones(init, n_in, n_out, Activation(Linear.forward, Linear.derivative)) for _ in range(n_out)])

    def set_activation(self: Neuronal_network, activation: Activation):
        """This method makes it possible to measure an activation function and its derivative for each neuron."""

        for couche in self.neurones:

            for neurone in couche:

                neurone.activation = activation

    def set_softmax(self, value=True):

        self.softmax = value