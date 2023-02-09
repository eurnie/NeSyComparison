import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset as TorchDataset
from typing import Callable, List, Iterable, Tuple
from pathlib import Path

root_path = Path(__file__).parent.parent.joinpath('data')

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

datasets = {
    "train": torchvision.datasets.MNIST(
        root=str(root_path), train=True, download=True, transform=transform
    ),
    "test": torchvision.datasets.MNIST(
        root=str(root_path), train=False, download=True, transform=transform
    )
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

class MNIST_Images(object):
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, item):
        return datasets[self.subset][int(item[0])][0]

def digits_to_number(digits: Iterable[int]) -> int:
    number = 0
    for d in digits:
        number *= 10
        number += d
    return number

def addition_with_only_one_x_value(dataset: str):
    """Returns a dataset for binary addition"""
    return NN_MNISTOperator(
        dataset_name=dataset
    )

class NN_MNISTOperator(TorchDataset):
    def __getitem__(self, index: int) -> Tuple[list, list, int]:
        label = self._get_label(index)
        l1 = self.dataset[self.data[index][0]][0][0]
        l2 = self.dataset[self.data[index][1]][0][0]       
        return torch.cat((l1, l2), 0), label

    def __init__(
        self,
        dataset_name: str
    ):
        super(NN_MNISTOperator, self).__init__()
        self.dataset_name = dataset_name
        self.dataset = datasets[self.dataset_name]
        if (self.dataset_name == "train"):
            self.data, self.labels = parse_data("../data/MNIST/processed/train.txt")
        elif (self.dataset_name == "test"):
            self.data, self.labels = parse_data("../data/MNIST/processed/test.txt")

    def _get_label(self, i: int):
        return self.labels[i]

    def __len__(self):
        return len(self.data)

MNIST_train = MNIST_Images("train")
MNIST_test = MNIST_Images("test")