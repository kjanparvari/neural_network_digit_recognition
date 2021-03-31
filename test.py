from neural_network import NeuralNetwork
from read_mnist import load_train_data, load_test_data
import numpy as np

if __name__ == '__main__':
    print("Loading Data ...")
    nn = NeuralNetwork.load()
    # nn.test(data=load_train_data())
    nn.test(data=load_test_data())
