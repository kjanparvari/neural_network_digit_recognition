import numpy as np
import math


class NeuralNetwork:

    def __init__(self, *layers_sizes):
        self._layers_sizes = layers_sizes
        self._activation_function = "sigmoid"
        self._weights = {}
        self._bias_vectors = {}
        self._layers = {}
        self._activations = {}
        self._grad_b = {}
        self._grad_w = {}
        self._grad_a = {}
        for i in range(1, self.number_of_layers):
            cur_vector = np.zeros((self.layer_size(i), 1))
            self._bias_vectors[i] = cur_vector
            self._grad_b[i] = cur_vector.copy()
            # print(f"b{i}: {cur_vector.shape}")
        for i in range(self.number_of_layers - 1):
            cur_size = self.layer_size(i)
            nxt_size = self.layer_size(i + 1)
            self._weights[i + 1] = np.random.rand(nxt_size, cur_size)
            self._grad_w[i + 1] = np.zeros((nxt_size, cur_size))
            # print(f"w{i + 1}: {self._weights[i + 1].shape}")

    def weights(self, layer_number: int):
        return self._weights[layer_number]

    def bias(self, layer_number: int):
        return self._bias_vectors[layer_number]

    def layer(self, layer_number: int):
        if layer_number == -1:
            return self._layers[self.number_of_layers - 1]
        return self._layers[layer_number]

    def layer_size(self, layer_number: int):
        return self._layers_sizes[layer_number]

    @property
    def number_of_layers(self):
        return len(self._layers_sizes)

    def activation(self, layer_number: int):
        if layer_number <= -1:
            return self._activations[self.number_of_layers + layer_number]
        return self._activations[layer_number]

    @staticmethod
    def cost(output, label) -> float:
        _sum = 0
        for j in range(len(label)):
            _sum += (label[j] - output[j]) ** 2
        return _sum

    def train(self):
        pass

    def train_batch(self, samples):
        for vector, label in samples:
            self._layers[0] = vector.copy()
            self._activations[0] = vector.copy()
            self.forward_propagate()

        # print(self.cost)

    def accuracy(self, samples: list):
        corrects = 0
        for vector, label in samples:
            self._layers[0] = vector.copy()
            self._activations[0] = vector.copy()
            self.forward_propagate()
            output = self.activation(-1)
            if output.argmax() == label.argmax():
                corrects += 1
        accuracy = corrects / len(samples)
        print(f"accuracy: {round(accuracy * 100, 3)} %")
        return accuracy

    def back_propagate(self):
        pass

    def change_weights(self):
        pass

    def forward_propagate(self):
        for layer_number in range(1, self.number_of_layers):
            result_vector = self.compute_layer(layer_number)
            self._layers[layer_number] = result_vector.copy()
            result_vector = self.activate(layer_number)
            self._activations[layer_number] = result_vector.copy()

    def compute_layer(self, layer_number):
        return (self.weights(layer_number) @ self.activation(layer_number - 1)) + self.bias(layer_number)

    def activate(self, layer_number):
        if self._activation_function == "sigmoid":
            return self.matrix_sigmoid(self.layer(layer_number))

    def activate_deriv(self, layer_number):
        if self._activation_function == "sigmoid":
            ones = np.ones((self.layer_size(layer_number), 1))
            vec = self.activation(layer_number).copy()
            return np.multiply(vec, ones - vec)

    def compute_grads(self):
        pass

    @classmethod
    def matrix_sigmoid(cls, matrix):
        func = np.vectorize(cls.sigmoid)
        return func(matrix)

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def sigmoid_derivative(x: float) -> float:
        pass
