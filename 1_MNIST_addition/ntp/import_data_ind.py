import os
import torchvision
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path

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

def parse_data(filename, start, end):
    with open(filename) as f:
        entries = f.readlines()

    if (end >= len(entries)):
        end = len(entries)

    dataset = []
    labels = []

    for i in range(start, end):
        index_digit_1 = int(entries[i].split(" ")[0])
        index_digit_2 = int(entries[i].split(" ")[1])
        sum = int(entries[i].split(" ")[2])

        new_entry = []
        new_entry.append(index_digit_1)
        new_entry.append(index_digit_2)
        dataset.append(new_entry)

        labels.append(sum)
        
    return dataset, labels

def create_data_files(dataset_name, size_val):
    split_index = round(size_val * 30000)

    # if dataset_name == "train":
    #     data, labels = parse_data("../data/MNIST/processed/train.txt", split_index, 30000)
    # elif dataset_name == "dev":
    #     data, labels = parse_data("../data/MNIST/processed/train.txt", 0, split_index)
    # elif dataset_name == "test":
    #     data, labels = parse_data("../data/MNIST/processed/test.txt", 0, 5000)

    if dataset_name == "train":
        data, labels = parse_data("../data/MNIST/processed/train.txt", 0, 100)
    elif dataset_name == "dev":
        data, labels = parse_data("../data/MNIST/processed/train.txt", 0, 100)
    elif dataset_name == "test":
        data, labels = parse_data("../data/MNIST/processed/test.txt", 0, 100)

    filename = "data/MNIST_ind/{}.tsv".format(dataset_name)
    if os.path.exists(filename):
        os.remove(filename)

    for i, elem in enumerate(data):
        with open(filename, "a+") as f:
                f.write(str(elem[0]))
                f.write(" ")
                f.write(str(labels[i]))
                f.write(" ")
                f.write(str(elem[1]))
                f.write(" ")
                f.write("\n")

create_data_files("train", 0.1)
create_data_files("test", 0.1)
create_data_files("dev", 0.1)