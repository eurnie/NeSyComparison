import tensorflow as tf
import torchvision
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path

DATA_ROOT = Path(__file__).parent.parent.joinpath('data')

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

datasets_mnist = {
    "train": torchvision.datasets.MNIST(
        root=str(DATA_ROOT), train=True, download=True, transform=transform
    ),
    "test": torchvision.datasets.MNIST(
        root=str(DATA_ROOT), train=False, download=True, transform=transform
    ),
}

datasets_fashion_mnist = {
    "train": torchvision.datasets.FashionMNIST(
        root=str(DATA_ROOT), train=True, download=True, transform=transform
    ),
    "test": torchvision.datasets.FashionMNIST(
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

def get_mnist_data_as_numpy(dataset):
    """Returns numpy arrays of images and labels"""
    if dataset == "MNIST":
        datasets = datasets_mnist
    elif dataset == "FashionMNIST":
        datasets = datasets_fashion_mnist

    img_train = datasets["train"].data.numpy()
    img_train = img_train[:, :, :, np.newaxis]
    img_train = img_train/255.0
    label_train = datasets["train"].targets.numpy()
    img_test = datasets["test"].data.numpy()
    img_test = img_test/255.0
    img_test = img_test[:, :, :, np.newaxis]
    label_test = datasets["test"].targets.numpy()
    
    return img_train, label_train, img_test, label_test

# def get_mnist_data_as_numpy_original():
#     """Returns numpy arrays of images and labels"""
#     mnist = tf.keras.datasets.mnist
#     (img_train, label_train), (img_test, label_test) = mnist.load_data()
#     img_train, img_test = img_train/255.0, img_test/255.0
#     img_train = img_train[...,tf.newaxis]
#     img_test = img_test[...,tf.newaxis]
#     return img_train,label_train, img_test,label_test

def get_mnist_op_dataset(dataset, batch_size):
    """Returns tf.data.Dataset instance for an operation with the numbers of the mnist dataset.
    Iterating over it, we get (image_x1,...,image_xn,label) batches
    such that op(image_x1,...,image_xn)= label.
    """
    img_train, _, img_test, _ = get_mnist_data_as_numpy(dataset)
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

def get_mnist_op_dataset_val(dataset, size_val, batch_size):
    img_train, _, img_test, _ = get_mnist_data_as_numpy(dataset)
    if dataset == "MNIST":
        train_data_processed, train_labels = parse_data("../data/MNIST/processed/train.txt")
        test_data_processed, label_result_test = parse_data("../data/MNIST/processed/test.txt")
    elif dataset == "FashionMNIST":
        train_data_processed, train_labels = parse_data("../data/FashionMNIST/processed/train.txt")
        test_data_processed, label_result_test = parse_data("../data/FashionMNIST/processed/test.txt")
    
    split_index = round(size_val * 30000)
    label_result_val = train_labels[:split_index]
    label_result_train = train_labels[split_index:]

    img_per_operand_val_1 = [img_train[train_data_processed[i][0]] for i in range(0, split_index)]
    img_per_operand_val_2 = [img_train[train_data_processed[i][1]] for i in range(0, split_index)]
    img_per_operand_val = [img_per_operand_val_1, img_per_operand_val_2]

    img_per_operand_train_1 = [img_train[train_data_processed[i][0]] for i in range(split_index, len(train_data_processed))]
    img_per_operand_train_2 = [img_train[train_data_processed[i][1]] for i in range(split_index, len(train_data_processed))]
    img_per_operand_train = [img_per_operand_train_1, img_per_operand_train_2]

    img_per_operand_test_1 = [img_test[i[0]] for i in test_data_processed]
    img_per_operand_test_2 = [img_test[i[1]] for i in test_data_processed]
    img_per_operand_test = [img_per_operand_test_1, img_per_operand_test_2]

    ds_train = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_train)+(label_result_train,)).batch(batch_size)
    ds_val = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_val)+(label_result_val,)).batch(1)
    ds_test = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_test)+(label_result_test,)).batch(1)

    return ds_train, ds_val, ds_test

def get_mnist_op_dataset_k_fold(dataset):
    img_train, _, _, _ = get_mnist_data_as_numpy(dataset)
    if dataset == "MNIST":
        train_data_processed, label_result_train = parse_data("../data/MNIST/processed/train.txt")
    elif dataset == "FashionMNIST":
        train_data_processed, label_result_train = parse_data("../data/FashionMNIST/processed/train.txt")

    img_per_operand_train_1 = [img_train[i[0]] for i in train_data_processed]
    img_per_operand_train_2 = [img_train[i[1]] for i in train_data_processed]
    img_per_operand_train = [img_per_operand_train_1, img_per_operand_train_2]

    return img_per_operand_train, label_result_train

def get_mnist_op_dataset_log_iter(dataset, batch_size, log_iter):
    """Returns tf.data.Dataset instance for an operation with the numbers of the mnist dataset.
    Iterating over it, we get (image_x1,...,image_xn,label) batches
    such that op(image_x1,...,image_xn)= label.
    """   
    img_train, _, _, _ = get_mnist_data_as_numpy(dataset)
    if dataset == "MNIST":
        train_data_processed, label_result_train = parse_data("../data/MNIST/processed/train.txt")
    elif dataset == "FashionMNIST":
        train_data_processed, label_result_train = parse_data("../data/FashionMNIST/processed/train.txt")

    img_per_operand_train_1 = [img_train[i[0]] for i in train_data_processed]
    img_per_operand_train_2 = [img_train[i[1]] for i in train_data_processed]
    ds_train_list = []

    for i in range(0, len(img_per_operand_train_1)+log_iter, log_iter):
        ds_train_list.append(tf.data.Dataset.from_tensor_slices(tuple([img_per_operand_train_1[i:i+log_iter], 
            img_per_operand_train_2[i:i+log_iter]])+(label_result_train[i:i+log_iter],)).batch(batch_size))

    return ds_train_list

# def get_mnist_op_dataset_original(
#         count_train,
#         count_test,
#         buffer_size,
#         batch_size,
#         n_operands=2,
#         op=lambda args: args[0]+args[1]):
#     """Returns tf.data.Dataset instance for an operation with the numbers of the mnist dataset.
#     Iterating over it, we get (image_x1,...,image_xn,label) batches
#     such that op(image_x1,...,image_xn)= label.

#     Args:
#         n_operands: The number of sets of images to return, 
#             that is the number of operands to the operation.
#         op: Operation used to produce the label. 
#             The lambda arguments must be a list from which we can index each operand. 
#             Example: lambda args: args[0] + args[1]
#     """
#     if count_train*n_operands > 60000:
#         raise ValueError("The MNIST dataset comes with 60000 training examples. \
#             Cannot fetch %i examples for each %i operands for training." %(count_train,n_operands))
#     if count_test*n_operands > 10000:
#         raise ValueError("The MNIST dataset comes with 10000 test examples. \
#             Cannot fetch %i examples for each %i operands for testing." %(count_test,n_operands))
    
#     img_train,label_train,img_test,label_test = get_mnist_data_as_numpy_original()
    
#     img_per_operand_train = [img_train[i*count_train:i*count_train+count_train] for i in range(n_operands)]
#     label_per_operand_train = [label_train[i*count_train:i*count_train+count_train] for i in range(n_operands)]
#     label_result_train = np.apply_along_axis(op,0,label_per_operand_train)
#     img_per_operand_test = [img_test[i*count_test:i*count_test+count_test] for i in range(n_operands)]
#     label_per_operand_test = [label_test[i*count_test:i*count_test+count_test] for i in range(n_operands)]
#     label_result_test = np.apply_along_axis(op,0,label_per_operand_test)
    
#     ds_train = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_train)+(label_result_train,))\
#             .take(count_train).shuffle(buffer_size).batch(batch_size)
#     ds_test = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_test)+(label_result_test,))\
#             .take(count_test).shuffle(buffer_size).batch(batch_size)
    
#     return ds_train, ds_test