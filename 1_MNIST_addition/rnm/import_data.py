import numpy as np
import tensorflow as tf
import tensorflow as tf
import torchvision
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_ROOT = Path(__file__).parent.parent.joinpath('data')

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

datasets = {
    "train": torchvision.datasets.MNIST(
        root=str(DATA_ROOT), train=True, download=True, transform=transform
    ),
    "test": torchvision.datasets.MNIST(
        root=str(DATA_ROOT), train=False, download=True, transform=transform
    ),
}

def parse_data(filename):
    with open(filename) as f:
        entries = f.readlines()

    dataset = []
    labels = []

    for entry in entries:
        index_digit_1 = int(entry.split(" ")[0])
        index_digit_2 = int(entry.split(" ")[1])
        sum = int(entry.split(" ")[2])

        new_entry = []
        new_entry.append(index_digit_1)
        new_entry.append(index_digit_2)
        dataset.append(new_entry)

        labels.append(sum)
        
    return dataset, labels

def get_mnist_data_as_numpy():
    """Returns numpy arrays of images and labels"""
    img_train = datasets["train"].data.numpy()
    label_train = datasets["train"].targets.numpy()
    img_test = datasets["test"].data.numpy()
    label_test = datasets["test"].targets.numpy()
    
    return img_train, label_train, img_test, label_test

def get_mnist_op_dataset(batch_size):
    """Returns tf.data.Dataset instance for an operation with the numbers of the mnist dataset.
    Iterating over it, we get (image_x1,...,image_xn,label) batches
    such that op(image_x1,...,image_xn)= label.
    """
    img_train, _, img_test, _ = get_mnist_data_as_numpy()
    train_data_processed, label_result_train = parse_data("../data/MNIST/processed/train.txt")
    test_data_processed, label_result_test = parse_data("../data/MNIST/processed/test.txt")

    img_per_operand_train_1 = [img_train[i[0]] for i in train_data_processed]
    img_per_operand_train_2 = [img_train[i[1]] for i in train_data_processed]
    img_per_operand_train = [img_per_operand_train_1, img_per_operand_train_2]

    img_per_operand_test_1 = [img_test[i[0]] for i in test_data_processed]
    img_per_operand_test_2 = [img_test[i[1]] for i in test_data_processed]
    img_per_operand_test = [img_per_operand_test_1, img_per_operand_test_2]

    ds_train = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_train)+(label_result_train,)).batch(batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_test)+(label_result_test,)).batch(1)

    return ds_train, ds_test

def mnist_data():
    x_train, _, x_test, _ = get_mnist_data_as_numpy()
    train_data_processed, label_result_train = parse_data("../data/MNIST/processed/train.txt")
    test_data_processed, label_result_test = parse_data("../data/MNIST/processed/test.txt")
    
    def _inner(x, data, label):
        x_new = []
        for I in x:
            I = I / 255
            # I = img.rotate(I, float(np.random.rand() * 90), reshape=False)
            # I = I + 0.3 * np.random.randn(28,28)
            x_new.append(I)
        x = np.reshape(x_new, [-1, 28*28])

        # print("!!!!!!")
        # print(len(x))

        # links = np.zeros([len(x),len(x)])
        # for i,y_i in enumerate(y):
        #     for j, y_j in enumerate(y):

        #         if y_i == y_j + 1:
        #             # if np.random.rand() < 0.9:
        #                 links[i,j] = 1
        # links = np.reshape(links, [1, -1])

        # follows = np.zeros([10,10])
        # for i in range(10):
        #     for j in range(10):
        #         if i == j + 1:
        #             follows[i,j] = 1
        # follows = np.reshape(follows, [1, -1])

        # y = np.eye(10)[y]
        # digit = np.reshape(y, [1, -1])

        # hb = np.concatenate((digit, links, follows), axis=1)
        return (x, data, label)

    return _inner(x_train, train_data_processed, label_result_train), _inner(x_test, test_data_processed, label_result_test)