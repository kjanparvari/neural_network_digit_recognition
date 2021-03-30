from read_mnist import load_sample_train_set
from neural_network import NeuralNetwork

if __name__ == '__main__':
    # save_sample_train_set()
    samples = load_sample_train_set()
    network = NeuralNetwork(784, 16, 16, 10)
    network.train(samples)
    exit(1)
