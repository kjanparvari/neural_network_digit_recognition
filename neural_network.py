import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm


class NeuralNetwork:
    _dist_address: str = "./dist/neural_network.kj"

    def __init__(self, *layers_sizes, learning_rate: float = 1.0, activation_function: str = "sigmoid",
                 epoch_number: int = 20, batch_size: int = 10):
        self._layers_sizes: tuple = layers_sizes
        self._learning_rate: float = learning_rate
        self._activation_function: str = str.lower(activation_function).strip()
        self._leaky_relu_parameter = 0.01
        self._epoch_number: int = epoch_number
        self._batch_size: int = batch_size

        self._weights: dict = {}
        self._bias_vectors: dict = {}
        self._layers: dict = {}
        self._activations: dict = {}
        self._grad_b: dict = {}
        self._grad_w: dict = {}
        self._grad_a: dict = {}
        self._costs: list = []

        for layer_number in range(1, self.number_of_layers):
            cur_size: int = self.layer_size(layer_number)
            prv_size: int = self.layer_size(layer_number - 1)
            self._bias_vectors[layer_number] = np.zeros((self.layer_size(layer_number), 1))
            # print(f"b{layer_number}: {cur_vector.shape}")
            self._weights[layer_number] = np.random.rand(cur_size, prv_size)
            # print(f"w{layer_number + 1}: {self._weights[layer_number + 1].shape}")

    def weights(self, layer_number: int) -> np.ndarray:
        return self._weights[layer_number]

    def bias(self, layer_number: int) -> np.ndarray:
        return self._bias_vectors[layer_number]

    def layer(self, layer_number: int) -> np.ndarray:
        if layer_number <= -1:
            return self._layers[self.number_of_layers + layer_number]
        return self._layers[layer_number]

    def layer_size(self, layer_number: int) -> int:
        return self._layers_sizes[layer_number]

    def weights_grad(self, layer_number: int) -> np.ndarray:
        if layer_number <= -1:
            return self._grad_w[self.number_of_layers + layer_number]
        return self._grad_w[layer_number]

    def bias_grad(self, layer_number: int) -> np.ndarray:
        if layer_number <= -1:
            return self._grad_b[self.number_of_layers + layer_number]
        return self._grad_b[layer_number]

    def activation_grad(self, layer_number: int) -> np.ndarray:
        if layer_number <= -1:
            return self._grad_a[self.number_of_layers + layer_number]
        return self._grad_a[layer_number]

    @property
    def number_of_layers(self) -> int:
        return len(self._layers_sizes)

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    def activation(self, layer_number: int) -> np.ndarray:
        if layer_number <= -1:
            return self._activations[self.number_of_layers + layer_number]
        return self._activations[layer_number]

    @property
    def epoch_number(self) -> int:
        return self._epoch_number

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @staticmethod
    def cost(output: np.ndarray, label: np.ndarray) -> float:
        _sum = 0
        for j in range(len(label)):
            _sum += (label[j] - output[j]) ** 2
        return _sum

    def train(self, data_set: list):
        number_of_batches = math.ceil(len(data_set) / self.batch_size)
        with tqdm(total=number_of_batches * self.epoch_number) as bar:
            for epoch_number in range(self.epoch_number):
                for batch_number in range(number_of_batches):
                    bar.update(1)
                    batch = data_set[batch_number * self.batch_size: (batch_number + 1) * self.batch_size].copy()
                    self.train_batch(batch)
        print("Trained!.")
        self.test(data_set)
        self.save()
        print("file is in ./dist")
        plt.plot(self._costs)
        plt.show()

    def train_batch(self, batch: list):
        for vector, label in batch:
            self._layers[0] = vector.copy()
            self._activations[0] = vector.copy()
            self.forward_propagate()
            self._costs.append(self.cost(self.activation(-1), label))
            self.back_propagate(label)
            self.mean_grads(len(batch))
        self.change_weights()

    def mean_grads(self, batch_size):
        for layer_number in range(1, self.number_of_layers):
            self._grad_w[layer_number] = (1.0 / batch_size) * self._grad_w[layer_number]
            self._grad_b[layer_number] = (1.0 / batch_size) * self._grad_b[layer_number]

    def test(self, samples: list):
        corrects = 0
        with tqdm(total=len(samples)) as bar:
            for vector, label in samples:
                bar.update(1)
                self._layers[0] = vector.copy()
                self._activations[0] = vector.copy()
                self.forward_propagate()
                output = self.activation(-1)
                if output.argmax() == label.argmax():
                    corrects += 1
        accuracy = corrects / len(samples)
        print(f"accuracy: {round(accuracy * 100, 3)} %")
        return accuracy

    def back_propagate(self, label):
        # initializing grad vectors to zero
        for layer_number in range(1, self.number_of_layers):
            cur_size = self.layer_size(layer_number)
            prv_size = self.layer_size(layer_number - 1)
            self._grad_b[layer_number] = np.zeros((self.layer_size(layer_number), 1))
            self._grad_w[layer_number] = np.zeros((cur_size, prv_size))
            self._grad_a = dict()

        last_layer = self.number_of_layers - 1
        self._grad_a[last_layer] = 2 * (self.activation(last_layer) - label)

        for layer_number in range(last_layer, 0, -1):
            activation_derivative = self.activation_derivative(layer_number)

            self._grad_w[layer_number] += (self.activation_grad(layer_number) * activation_derivative) @ (
                self.activation(layer_number - 1).transpose())

            self._grad_b[layer_number] += (self.activation_grad(layer_number) * activation_derivative)

            if layer_number > 1:
                self._grad_a[layer_number - 1] = self.weights(layer_number).transpose() @ (
                        activation_derivative * self.activation_derivative(layer_number))

    def change_weights(self):
        for layer_number in range(1, self.number_of_layers):
            self._weights[layer_number] -= self.learning_rate * self._grad_w[layer_number]
            self._bias_vectors[layer_number] -= self.learning_rate * self._grad_b[layer_number]

    def forward_propagate(self):
        for layer_number in range(1, self.number_of_layers):
            result_vector = self.compute_layer(layer_number)
            self._layers[layer_number] = result_vector.copy()
            result_vector = self.activate(layer_number)
            self._activations[layer_number] = result_vector.copy()

    def compute_layer(self, layer_number) -> np.ndarray:
        return (self.weights(layer_number) @ self.activation(layer_number - 1)) + self.bias(layer_number)

    def activate(self, layer_number) -> np.ndarray:
        if self._activation_function == "sigmoid":
            func = np.vectorize(self.sigmoid)
        elif self._activation_function == "relu":
            func = np.vectorize(self.relu)
        elif self._activation_function == "leaky relu":
            func = np.vectorize(self.leaky_relu)
        else:
            func = np.vectorize(self.sigmoid)
        return func(self.layer(layer_number))

    def activation_derivative(self, layer_number) -> np.ndarray:
        vec = self.activation(layer_number).copy()
        if self._activation_function == "sigmoid":
            ones = np.ones((self.layer_size(layer_number), 1))
            return vec * (ones - vec)
        elif self._activation_function == "relu":
            return np.ones((self.layer_size(layer_number), 1))
        elif self._activation_function == "leaky relu":
            filtered_lst = []
            for element in vec:
                if element[0] >= 0:
                    filtered_lst.append([1])
                else:
                    filtered_lst.append([self._leaky_relu_parameter])
            return np.array(filtered_lst, copy=True)

    def save(self):
        import pickle
        with open(self._dist_address, "wb") as f:
            pickle.dump(self, f)
            f.close()

    @classmethod
    def load(cls):
        import pickle
        nn: NeuralNetwork
        with open(cls._dist_address, "rb") as f:
            nn = pickle.load(f)
            f.close()
        return nn

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def relu(x: float) -> float:
        if x >= 0:
            return x
        else:
            return 0

    def leaky_relu(self, x: float) -> float:
        if x >= 0:
            return x
        else:
            return self._leaky_relu_parameter * x
