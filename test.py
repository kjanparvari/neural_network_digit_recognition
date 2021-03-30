from neural_network import NeuralNetwork
from read_mnist import load_sample_train_set


if __name__ == '__main__':
    network = NeuralNetwork.load()
    samples = load_sample_train_set()
    network.test(samples)
    exit(1)
