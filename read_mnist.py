import numpy as np


def load_test_data():
    test_images = open('t10k-images.idx3-ubyte', 'rb')
    test_images.seek(4)
    num_of_test_images = int.from_bytes(test_images.read(4), 'big')
    test_images.seek(16)
    test_labels = open('t10k-labels.idx1-ubyte', 'rb')
    test_labels.seek(8)

    test_set = []

    for n in range(num_of_test_images):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i, 0] = int.from_bytes(test_images.read(1), 'big') / 256

        label_value = int.from_bytes(test_labels.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1

        test_set.append((image, label))
    return test_set


def load_train_data():
    train_images_file = open('train-images.idx3-ubyte', 'rb')
    train_images_file.seek(4)
    num_of_train_images = int.from_bytes(train_images_file.read(4), 'big')
    train_images_file.seek(16)
    train_labels_file = open('train-labels.idx1-ubyte', 'rb')
    train_labels_file.seek(8)

    train_set = []

    for n in range(num_of_train_images):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i, 0] = int.from_bytes(train_images_file.read(1), 'big') / 256

        label_value = int.from_bytes(train_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1

        train_set.append((image, label))
    return train_set


def load_adversarial_test_data():
    test_images = open('t10k-images.idx3-ubyte', 'rb')
    test_images.seek(4)
    num_of_test_images = int.from_bytes(test_images.read(4), 'big')
    test_images.seek(16)
    test_labels = open('t10k-labels.idx1-ubyte', 'rb')
    test_labels.seek(8)

    test_set = []

    for n in range(num_of_test_images):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i, 0] = int.from_bytes(test_images.read(1), 'big') / 256

        for i in range(0, 28):
            for j in range(4):
                image = np.delete(image, i * 28 - 1, 0)
                image = np.insert(image, i * 28, np.array(0), 0)

        label_value = int.from_bytes(test_labels.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1

        test_set.append((image, label))
    return test_set
