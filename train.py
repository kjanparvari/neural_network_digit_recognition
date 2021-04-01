from neural_network import NeuralNetwork
from read_mnist import load_train_data

if __name__ == '__main__':
    print("Loading Data ...")
    # lr=1.0 fro sigmoid - lr=0.001 for leaky relu -
    nn = NeuralNetwork(784, 16, 16, 10, epoch_number=5, batch_size=50, learning_rate=1, method="sigmoid")
    nn.train(data=load_train_data())
