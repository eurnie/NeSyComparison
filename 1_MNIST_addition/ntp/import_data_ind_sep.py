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

def create_data_files(dataset_name, size_val):
    split_index = round(size_val * 60000)

    filename = "data/MNIST_ind_sep/{}.tsv".format(dataset_name)
    if os.path.exists(filename):
        os.remove(filename)

    if dataset_name == "train":
        start = split_index
        end = len(datasets["train"])
        data = datasets["train"]
    elif dataset_name == "dev":
        start = 0
        end = split_index
        data = datasets["train"]
    elif dataset_name == "test":
        start = 0
        end = len(datasets["test"])
        data = datasets["test"]

    # if dataset_name == "train":
    #     start = split_index
    #     end = split_index + 5000
    #     data = datasets["train"]
    # elif dataset_name == "dev":
    #     start = 0
    #     end = 10
    #     data = datasets["train"]
    # elif dataset_name == "test":
    #     start = 0
    #     end = 10
    #     data = datasets["test"]

    for i in range(start, end):
        # digit = data[i][0]
        label = data[i][1]

        with open(filename, "a+") as f:
                f.write(str(i))
                f.write(" ")
                f.write("digit")
                f.write(" ")
                f.write(str(label))
                f.write(" ")
                f.write("\n")

create_data_files("train", 0.1)
create_data_files("test", 0.1)
create_data_files("dev", 0.1)