import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm


class NeuralNetwork:

    def __init__(self, *layer_sizes, epoch_number, batch_size, learning_rate, method, use_momentum=False):
        self.layer_sizes = layer_sizes
        self.epoch_number = epoch_number
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.leaky_parameter = 0.01
        self.method = method
        self.use_momentum = use_momentum
        self.weights = {}
        self.biases = {}
        self.costs = []

        self.decay_factor = 0.9
        # self.corrects = np.zeros(epoch_number)
        if self.method != "leaky relu":
            for layer_number in range(1, self.network_size):
                self.weights[layer_number] = np.random.randn(layer_sizes[layer_number], layer_sizes[layer_number - 1])
                self.biases[layer_number] = np.zeros((layer_sizes[layer_number], 1))
        else:
            self.weights[1], self.biases[1] = np.random.normal(0, 2 / 784, size=(16, 784)), np.zeros((16, 1))
            for layer_number in range(2, self.network_size):
                self.weights[layer_number], self.biases[layer_number] = self.rai(layer_sizes[layer_number - 1],
                                                                                 layer_sizes[layer_number])

    @property
    def network_size(self):
        return len(self.layer_sizes)

    def active(self, array, method=None):
        if method is None:
            method = self.method
        # method = self.method
        if method == "sigmoid":
            return 1 / (1 + np.exp(-array))
        elif method == "relu":
            return np.maximum(0, array)
        elif method == "leaky relu":
            y1 = ((array > 0) * array)
            y2 = ((array <= 0) * array * self.leaky_parameter)
            return y1 + y2

    def active_deriv(self, array, method=None):
        if method is None:
            method = self.method
        # method = self.method
        if method == "sigmoid":
            return array * (1 - array)
        elif method == "relu":
            dx = np.ones_like(array)
            dx[array <= 0] = 0
            return dx
        elif method == "leaky relu":
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
                if layer_number == self.network_size - 1:
                    layers[layer_number] = self.active(
                        np.matmul(self.weights[layer_number], layers[layer_number - 1]) + self.biases[layer_number],
                        method="sigmoid")
                else:
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
        prev_g_weights = {}
        prev_g_biases = {}
        for layer_number in range(1, self.network_size):
            prev_g_weights[layer_number] = np.zeros(
                (self.layer_sizes[layer_number], self.layer_sizes[layer_number - 1]))
            prev_g_biases[layer_number] = np.zeros((self.layer_sizes[layer_number], 1))
        costs = []
        batch_cost = 0
        number_of_batches = math.ceil(len(data) / self.batch_size)
        print("Training ...")
        bar = tqdm(total=number_of_batches * self.epoch_number * self.batch_size)
        for epoch_number in range(self.epoch_number):
            np.random.shuffle(data)
            for batch_number in range(number_of_batches):
                # init gradients to zero
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
                        if layer_number == self.network_size - 1:
                            layers[layer_number] = self.active(
                                np.matmul(self.weights[layer_number], layers[layer_number - 1]) + self.biases[
                                    layer_number], method="sigmoid")
                        else:
                            layers[layer_number] = self.active(
                                np.matmul(self.weights[layer_number], layers[layer_number - 1]) + self.biases[
                                    layer_number])
                    # if layers[self.network_size - 1].argmax() == label.argmax():
                    # self.corrects[epoch_number] += 1
                    batch_cost += ((layers[self.network_size - 1] - label) ** 2).mean(axis=None)
                    g_acts[self.network_size - 1] = 2 * (layers[self.network_size - 1] - label)
                    for layer_number in range(self.network_size - 1, 0, -1):
                        if layer_number == self.network_size - 1:
                            act_deriv = self.active_deriv(layers[layer_number], method="sigmoid")
                        else:
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
                    if self.use_momentum:
                        gw = self.decay_factor * prev_g_weights[layer_number] - (
                                1.0 - self.decay_factor) * self.learning_rate * g_weights[
                                 layer_number] / self.batch_size
                        gb = self.decay_factor * prev_g_biases[layer_number] - (
                                1.0 - self.decay_factor) * self.learning_rate * g_biases[layer_number] / self.batch_size
                        prev_g_weights[layer_number] = gw
                        prev_g_biases[layer_number] = gb
                    else:
                        gw = self.learning_rate * g_weights[layer_number] / self.batch_size
                        gb = self.learning_rate * g_biases[layer_number] / self.batch_size

                    self.weights[layer_number] += gw
                    self.biases[layer_number] += gb

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
    def rai(fan_in, fan_out):
        v = np.random.randn(fan_out, fan_in + 1) * 0.6007 / fan_in ** 0.5
        for j in range(fan_out):
            k = np.random.randint(0, high=fan_in + 1)
            v[j, k] = np.random.beta(2, 1)
        w = v[:, :-1]
        b = np.reshape(v[:, -1], (fan_out, 1))
        return w.astype(np.float32), b.astype(np.float32)

    @staticmethod
    def load():
        import pickle
        nn: NeuralNetwork
        with open("meta.kj", "rb") as f:
            nn = pickle.load(f)
        return nn
