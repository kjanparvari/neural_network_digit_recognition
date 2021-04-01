import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm


class NeuralNetwork:

    def __init__(self, *layer_sizes, epoch_number, batch_size, learning_rate, method):
        self.layer_sizes = layer_sizes
        self.epoch_number = epoch_number
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.leaky_parameter = 0.01
        self.method = method
        self.weights = {}
        self.biases = {}
        self.costs = []
        # self.corrects = np.zeros(epoch_number)

        for layer_number in range(1, self.network_size):
            self.weights[layer_number] = np.random.randn(layer_sizes[layer_number], layer_sizes[layer_number - 1])
            self.biases[layer_number] = np.zeros((layer_sizes[layer_number], 1))

    @property
    def network_size(self):
        return len(self.layer_sizes)

    def active(self, array):
        if self.method == "sigmoid":
            return 1 / (1 + np.exp(-array))
        elif self.method == "relu":
            return np.maximum(0, array)
        elif self.method == "leaky relu":
            y1 = ((array > 0) * array)
            y2 = ((array <= 0) * array * self.leaky_parameter)
            return y1 + y2

    def active_deriv(self, array):
        if self.method == "sigmoid":
            return array * (1 - array)
        elif self.method == "relu":
            dx = np.ones_like(array)
            dx[array <= 0] = 0
            return dx
        elif self.method == "leaky relu":
            dx = np.ones_like(array)
            dx[array < 0] = self.leaky_parameter
            return dx

    def test(self, data):
        bar = tqdm(total=len(data))
        corrects = 0
        for i in range(len(data)):
            bar.update(1)
            image, label = data[i]
            layers = {}
            for layer_number in range(1, self.network_size):
                layers[0] = image
                layers[layer_number] = self.active(
                    np.matmul(self.weights[layer_number], layers[layer_number - 1]) + self.biases[layer_number])
            if layers[self.network_size - 1].argmax() == label.argmax():
                corrects += 1
        bar.close()
        print(f"Accuracy: {round(100 * corrects / len(data), 3)}%")

    def train(self, data):
        g_weights = {}
        g_biases = {}
        g_acts = {}
        costs = []
        batch_cost = 0
        number_of_batches = math.ceil(len(data) / self.batch_size)
        bar = tqdm(total=number_of_batches * self.epoch_number * self.batch_size)
        for epoch_number in range(self.epoch_number):
            np.random.shuffle(data)
            for batch_number in range(number_of_batches):
                for layer_number in range(1, self.network_size):
                    g_weights[layer_number] = np.zeros(
                        (self.layer_sizes[layer_number], self.layer_sizes[layer_number - 1]))
                    g_biases[layer_number] = np.zeros((self.layer_sizes[layer_number], 1))
                for i in range(self.batch_size):
                    bar.update(1)
                    image, label = data[self.batch_size * batch_number + i]
                    layers = {}
                    for layer_number in range(1, self.network_size):
                        layers[0] = image
                        layers[layer_number] = self.active(
                            np.matmul(self.weights[layer_number], layers[layer_number - 1]) + self.biases[layer_number])
                    # if layers[self.network_size - 1].argmax() == label.argmax():
                    # self.corrects[epoch_number] += 1
                    batch_cost += ((layers[self.network_size - 1] - label) ** 2).mean(axis=None)
                    g_acts[self.network_size - 1] = 2 * (layers[self.network_size - 1] - label)
                    for layer_number in range(self.network_size - 1, 0, -1):
                        act_deriv = self.active_deriv(layers[layer_number])
                        # print(layers[layer_number], "\n\n")
                        # print(act_deriv, "\n----------------------\n")
                        tmp = np.multiply(act_deriv, g_acts[layer_number])
                        g_biases[layer_number] += tmp
                        # print(f"tmp: {tmp.shape} | layer: {layers[layer_number - 1].shape}")
                        g_weights[layer_number] += np.matmul(tmp, layers[layer_number - 1].transpose())
                        if layer_number > 1:
                            g_acts[layer_number - 1] = np.matmul(self.weights[layer_number].transpose(), tmp)
                for layer_number in range(1, self.network_size):
                    self.weights[layer_number] -= self.learning_rate * g_weights[layer_number] / self.batch_size
                    self.biases[layer_number] -= self.learning_rate * g_biases[layer_number] / self.batch_size
            costs.append(batch_cost / self.batch_size)
            batch_cost = 0
            # self.corrects[epoch_number] /= len(data)
        bar.close()
        print("Trained!")
        self.save()
        plt.plot(costs)
        plt.show()
        # self.test(data)
        # print(f"Accuracy: {round(100 * self.corrects[self.epoch_number - 1], 3)}%")

    def save(self):
        import pickle
        with open("meta.kj", "wb") as f:
            pickle.dump(self, f)
        print("Saved in meta.kj")

    @staticmethod
    def load():
        import pickle
        nn: NeuralNetwork
        with open("meta.kj", "rb") as f:
            nn = pickle.load(f)
        return nn
