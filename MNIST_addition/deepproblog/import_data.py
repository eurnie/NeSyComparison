import json
import itertools
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from torch.utils.data import Dataset as TorchDataset
from typing import Callable, List, Iterable, Tuple
from deepproblog.dataset import Dataset
from deepproblog.query import Query
from problog.logic import Term, list2term, Constant
from deepproblog.dataset import Dataset

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
        new_entry.append([index_digit_1])
        new_entry.append([index_digit_2])
        dataset.append(new_entry)

        labels.append(sum)
        
    return dataset, labels

def digits_to_number(digits: Iterable[int]) -> int:
    number = 0
    for d in digits:
        number *= 10
        number += d
    return number

class MNIST_Images(object):
    def __init__(self, dataset, subset):
        self.subset = subset
        self.dataset = dataset

    def __getitem__(self, item):
        if self.dataset == "mnist":
            if self.subset == "val":
                return datasets_mnist["train"][int(item[0]+2700)][0]
            else:
                return datasets_mnist[self.subset][int(item[0])][0]
        elif self.dataset == "fashion_mnist":
            if self.subset == "val":
                return datasets_fashion_mnist["train"][int(item[0]+2700)][0]
            else:
                return datasets_fashion_mnist[self.subset][int(item[0])][0]

def import_datasets(dataset, size_val):
    split_index = round(size_val * 30000)
    train_set = MNISTOperator(
        dataset=dataset,
        dataset_name="train",
        function_name="addition",
        operator=sum,
        size=1,
        arity=2,
        start_index=split_index,
        end_index=30000
    )
    val_set = MNISTOperator(
        dataset=dataset,
        dataset_name="train",
        function_name="addition",
        operator=sum,
        size=1,
        arity=2,
        start_index=0,
        end_index=split_index
    )
    test_set = MNISTOperator(
        dataset=dataset,
        dataset_name="test",
        function_name="addition",
        operator=sum,
        size=1,
        arity=2,
        start_index=0,
        end_index=100
    )
    return train_set, val_set, test_set

def import_datasets_kfold(dataset):
    train_set_list = []
    for i in range(0, 10):
        train_set = MNISTOperator(
            dataset=dataset,
            dataset_name="train",
            function_name="addition",
            operator=sum,
            size=1,
            arity=2,
            start_index=i*3000,
            end_index=(i+1)*3000
        )
        train_set_list.append(train_set)
    return train_set_list

class MNISTOperator(Dataset, TorchDataset):
    def __getitem__(self, index: int) -> Tuple[list, list, int]:
        l1, l2 = self.data[index]
        label = self._get_label(index)
        l1 = [self.dataset[x][0] for x in l1]
        l2 = [self.dataset[x][0] for x in l2]
        return l1, l2, label

    def __init__(
        self,
        dataset: str,
        dataset_name: str,
        function_name: str,
        operator: Callable[[List[int]], int],
        start_index=0,
        end_index=30001,
        size=1,
        arity=2,
    ):
        """Generic dataset for operator(img, img) style datasets.

        :param dataset_name: Dataset to use (train, val, test)
        :param function_name: Name of Problog function to query.
        :param operator: Operator to generate correct examples
        :param size: Size of numbers (number of digits)
        :param arity: Number of arguments for the operator
        """
        super(MNISTOperator, self).__init__()
        assert size >= 1
        assert arity >= 1
        self.dataset_name = dataset_name

        self.function_name = function_name
        self.operator = operator
        self.size = size
        self.arity = arity      

        if dataset == "mnist":
            self.dataset = datasets_mnist[self.dataset_name]
            if dataset_name == "train":
                self.data, self.labels = parse_data("../data/MNIST/processed/train.txt", start_index, end_index)
            elif dataset_name == "test":
                self.data, self.labels = parse_data("../data/MNIST/processed/test.txt", 0, 5000)
        elif dataset == "fashion_mnist":
            self.dataset = datasets_fashion_mnist[self.dataset_name]
            if dataset_name == "train":
                self.data, self.labels = parse_data("../data/FashionMNIST/processed/train.txt", start_index, end_index)
            elif dataset_name == "test":
                self.data, self.labels = parse_data("../data/FashionMNIST/processed/test.txt", 0, 5000)

    def to_file_repr(self, i):
        """Old file represenation dump. Not a very clear format as multi-digit arguments are not separated"""
        return f"{tuple(itertools.chain(*self.data[i]))}\t{self._get_label(i)}"

    def to_json(self):
        """
        Convert to JSON, for easy comparisons with other systems.

        Format is [EXAMPLE, ...]
        EXAMPLE :- [ARGS, expected_result]
        ARGS :- [MULTI_DIGIT_NUMBER, ...]
        MULTI_DIGIT_NUMBER :- [mnist_img_id, ...]
        """
        data = [(self.data[i], self._get_label(i)) for i in range(len(self))]
        return json.dumps(data)

    def to_query(self, i: int) -> Query:
        """Generate queries"""
        mnist_indices = self.data[i]
        expected_result = self._get_label(i)

        # Build substitution dictionary for the arguments
        subs = dict()
        var_names = []
        for i in range(self.arity):
            inner_vars = []
            for j in range(self.size):
                t = Term(f"p{i}_{j}")
                subs[t] = Term(
                    "tensor",
                    Term(
                        self.dataset_name,
                        Constant(mnist_indices[i][j]),
                    ),
                )
                inner_vars.append(t)
            var_names.append(inner_vars)

        # Build query
        if self.size == 1:
            return Query(
                Term(
                    self.function_name,
                    *(e[0] for e in var_names),
                    Constant(expected_result),
                ),
                subs,
            )
        else:
            return Query(
                Term(
                    self.function_name,
                    *(list2term(e) for e in var_names),
                    Constant(expected_result),
                ),
                subs,
            )

    def _get_label(self, i: int):
        return self.labels[i]

    def __len__(self):
        return len(self.data)

MNIST_train = MNIST_Images("mnist", "train")
MNIST_val = MNIST_Images("mnist", "val")
MNIST_test = MNIST_Images("mnist", "test")

FashionMNIST_train = MNIST_Images("fashion_mnist", "train")
FashionMNIST_val = MNIST_Images("fashion_mnist", "val")
FashionMNIST_test = MNIST_Images("fashion_mnist", "test")