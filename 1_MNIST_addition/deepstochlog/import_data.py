import random
import typing
import torchvision
from pathlib import Path
from typing import List, Tuple, Generator
from torchvision import transforms as transforms
from torchvision.datasets import MNIST
from deepstochlog.context import Context, ContextualizedTerm
from deepstochlog.term import Term, List
from typing import Sequence, Union

DATA_ROOT = Path(__file__).parent.parent.joinpath('data')

t1 = Term("t1")
t2 = Term("t2")
argument_sequence = List(t1, t2)

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

class SimpleAdditionDataset(Sequence):
    def __init__(self, dataset_name, start_index, end_index, digit_length=1, size: int = None):
        self.mnist_dataset = get_mnist_data(dataset_name)
        self.ct_term_dataset = []
        if (dataset_name == "train"):
            dataset, labels = parse_data("../data/MNIST/processed/train.txt", start_index, end_index)
        elif (dataset_name == "test"):
            dataset, labels = parse_data("../data/MNIST/processed/test.txt", start_index, end_index)
        for idx in range(0, len(dataset)):
            mnist_datapoint_1 = self.mnist_dataset[dataset[idx][0]][0]
            mnist_datapoint_2 = self.mnist_dataset[dataset[idx][1]][0]
            # digit_1 = mnist_datapoint_1[1]
            # digit_2 = mnist_datapoint_2[1]
            total_sum = labels[idx]
            addition_term = ContextualizedTerm(
                # load context with the tensors
                context=Context({t1: mnist_datapoint_1, t2: mnist_datapoint_2}),
                # create the term containing the sum and a list of tokens representing the tensors
                term=Term(
                    "addition",
                    Term(str(total_sum)),
                    argument_sequence,
                ),
                # meta=str(digit_1) + "+" + str(digit_2),
            )
            self.ct_term_dataset.append(addition_term)

    def __len__(self):
        return len(self.ct_term_dataset)

    def __getitem__(self, item: Union[int, slice]):
        if type(item) is slice:
            return (self[i] for i in range(*item.indices(len(self))))
        return self.ct_term_dataset[item]

def get_mnist_data(dataset) -> MNIST:
    return datasets[dataset]

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

def get_mnist_digits(
    train: bool, digits: List[int], output_names: bool = False
) -> Tuple[Generator, ...]:
    dataset = get_mnist_data(train)
    if not output_names:
        return tuple(
            (
                dataset[i][0]
                for i in (dataset.targets == digit).nonzero(as_tuple=True)[0]
            )
            for digit in digits
        )
    prefix = "train_" if train else "test_"
    return tuple(
        (prefix + str(i.item()) for i in (dataset.targets == digit).nonzero(as_tuple=True)[0])
        for digit in digits
    )

def split_train_dataset(dataset: typing.List, val_num_digits: int, train: bool):
    if train:
        return dataset[val_num_digits:]
    else:
        return dataset[:val_num_digits]

def get_next(idx: int, elements: typing.MutableSequence) -> Tuple[any, int]:
    if idx >= len(elements):
        idx = 0
        random.shuffle(elements)
    result = elements[idx]
    idx += 1
    return result, idx

def import_datasets(size_val):
    split_index = round(size_val * 30000)
    train_set = SimpleAdditionDataset("train", split_index, 30000)
    val_set = SimpleAdditionDataset("train", 0, split_index)
    test_set = SimpleAdditionDataset("test", 0, 5000)

    return train_set, val_set, test_set