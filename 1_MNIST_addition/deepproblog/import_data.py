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
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, item):
        if self.subset == "val":
            return datasets["train"][int(item[0]+2700)][0]
        else:
            return datasets[self.subset][int(item[0])][0]

class MNIST(Dataset):
    def __len__(self):
        return len(self.data)

    def to_query(self, i):
        l = Constant(self.data[i][1])
        return Query(
            Term("digit", Term("tensor", Term(self.dataset, Term("a"))), l),
            substitution={Term("a"): Constant(i)},
        )

    def __init__(self, dataset):
        self.dataset = dataset
        self.data = datasets[dataset]

def import_dataset_log_iter(n: int, dataset: str, log_iter=10):
    MNISTOperator_list = []

    if dataset == "train":
        for i in range(0, 30000, log_iter):
            MNISTOperator_list.append(MNISTOperator(
                                    dataset_name=dataset,
                                    function_name="addition" if n == 1 else "multi_addition",
                                    operator=sum,
                                    start_index=i,
                                    end_index=i+log_iter,
                                    size=n,
                                    arity=2,
        ))
        return MNISTOperator_list
    elif dataset == "test":
        return MNISTOperator(
            dataset_name=dataset,
            function_name="addition" if n == 1 else "multi_addition",
            operator=sum,
            size=n,
            arity=2
        )

def import_datasets(size_val):
    split_index = round(size_val * 30000)
    train_set = MNISTOperator(
        dataset_name="train",
        function_name="addition",
        operator=sum,
        size=1,
        arity=2,
        start_index=split_index,
        end_index=30000
    )
    val_set = MNISTOperator(
        dataset_name="train",
        function_name="addition",
        operator=sum,
        size=1,
        arity=2,
        start_index=0,
        end_index=split_index
    )
    test_set = MNISTOperator(
        dataset_name="test",
        function_name="addition",
        operator=sum,
        size=1,
        arity=2
    )
    return train_set, val_set, test_set

class MNISTOperator(Dataset, TorchDataset):
    def __getitem__(self, index: int) -> Tuple[list, list, int]:
        l1, l2 = self.data[index]
        label = self._get_label(index)
        l1 = [self.dataset[x][0] for x in l1]
        l2 = [self.dataset[x][0] for x in l2]
        return l1, l2, label

    def __init__(
        self,
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
        self.dataset = datasets[self.dataset_name]
        self.function_name = function_name
        self.operator = operator
        self.size = size
        self.arity = arity      

        if dataset_name == "train":
            self.data, self.labels = parse_data("../data/MNIST/processed/train.txt", start_index, end_index)
        elif dataset_name == "test":
            self.data, self.labels = parse_data("../data/MNIST/processed/test.txt", 0, 5000)

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

MNIST_train = MNIST_Images("train")
MNIST_val = MNIST_Images("val")
MNIST_test = MNIST_Images("test")