from read_mnist import get_train_set
import pickle
from neural_network import NeuralNetwork


def save_sample_train_set():
    train_set = get_train_set()
    _sample = train_set[0:100].copy()
    with open("sample_train_set.tr", 'wb') as f:
        pickle.dump(train_set, f)
        f.close()


def load_sample_train_set():
    return pickle.load(open("sample_train_set.tr", 'rb'))


if __name__ == '__main__':
    # save_sample_train_set()
    samples = load_sample_train_set()
    network = NeuralNetwork(784, 16, 16, 10)
    # network.train_batch(samples)
    network.accuracy(samples)
    # import numpy as np
    #
    # vec1 = np.array([[1] for i in range(5)])
    # vec2 = np.array([[2 for j in range(5)]])
    exit(1)
