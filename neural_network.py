import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm


class NeuralNetwork:

    def __init__(self, *layer_sizes, epoch_number, batch_size, learning_rate):
        self.layer_sizes = layer_sizes
        self.epoch_number = epoch_number
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weights = {}
        self.biases = {}
        # self.corrects = np.zeros(epoch_number)

        for layer_number in range(1, self.network_size):
            self.weights[layer_number] = np.random.randn(layer_sizes[layer_number], layer_sizes[layer_number - 1])
            self.biases[layer_number] = np.zeros((layer_sizes[layer_number], 1))

    @property
    def network_size(self):
        return len(self.layer_sizes)

    @staticmethod
    def sigmoid(array):
        return 1 / (1 + np.exp(-array))

    def test(self, data):
        bar = tqdm(total=len(data))
        corrects = 0
        for i in range(len(data)):
            bar.update(1)
            image, label = data[i]
            layers = {}
            for layer_number in range(1, self.network_size):
                layers[0] = image
                layers[layer_number] = self.sigmoid(
                    np.matmul(self.weights[layer_number], layers[layer_number - 1]) + self.biases[layer_number])
            if layers[self.network_size - 1].argmax() == label.argmax():
                corrects += 1
        bar.close()
        print(f"Accuracy: {round(100 * corrects / len(data), 3)}%")

    def train(self, data):
        g_weights = {}
        g_biases = {}
        g_acts = {}
        number_of_batches = math.ceil(len(data) / self.batch_size)
        bar = tqdm(total=number_of_batches * self.epoch_number * self.batch_size)
        for epoch_number in range(self.epoch_number):
            # print(f"epoch: {epoch_number}")
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
                        layers[layer_number] = self.sigmoid(
                            np.matmul(self.weights[layer_number], layers[layer_number - 1]) + self.biases[layer_number])
                    # if layers[self.network_size - 1].argmax() == label.argmax():
                    # self.corrects[epoch_number] += 1
                    g_acts[self.network_size - 1] = 2 * (layers[self.network_size - 1] - label)
                    for layer_number in range(self.network_size - 1, 0, -1):
                        act_deriv = layers[layer_number] * (1 - layers[layer_number])
                        tmp = np.multiply(act_deriv, g_acts[layer_number])
                        g_biases[layer_number] += tmp
                        # print(f"tmp: {tmp.shape} | layer: {layers[layer_number - 1].shape}")
                        g_weights[layer_number] += np.matmul(tmp, layers[layer_number - 1].transpose())
                        if layer_number > 1:
                            g_acts[layer_number - 1] = np.matmul(self.weights[layer_number].transpose(), tmp)
                for layer_number in range(1, self.network_size):
                    self.weights[layer_number] -= self.learning_rate * g_weights[layer_number] / self.batch_size
                    self.biases[layer_number] -= self.learning_rate * g_biases[layer_number] / self.batch_size
            # self.corrects[epoch_number] /= len(data)
        bar.close()
        print("Trained!")
        self.save()
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
