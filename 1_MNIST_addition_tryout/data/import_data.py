import itertools
import json
import torch
import random
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset as TorchDataset
from typing import Callable, List, Iterable, Tuple
from pathlib import Path

root_path = Path(__file__).parent

# transformation used in DeepProbLog experiments
transform_1 = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# transformation used in NeurASP experiments
transform_2 = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081, ))])

datasets = {
    "train": torchvision.datasets.MNIST(
        root=str(root_path), train=True, download=True, transform=transform_1
    ),
    "test": torchvision.datasets.MNIST(
        root=str(root_path), train=False, download=True, transform=transform_1
    )
}

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
        return datasets[self.subset][int(item[0])][0]

MNIST_train = MNIST_Images("train")
MNIST_test = MNIST_Images("test")

def addition(n: int, dataset: str, seed=None):
    """Returns a dataset for binary addition"""
    return GeneralMNISTOperator(
        dataset_name=dataset,
        function_name="addition" if n == 1 else "multi_addition",
        operator=sum,
        size=n,
        arity=2,
        seed=seed,
    )

def addition_with_only_one_x_value(n: int, dataset: str, seed=None):
    """Returns a dataset for binary addition"""
    return GeneralMNISTOperator(
        dataset_name=dataset,
        function_name="addition" if n == 1 else "multi_addition",
        operator=sum,
        size=n,
        arity=2,
        seed=seed,
        only_one_x_value=True
    )

class GeneralMNISTOperator(TorchDataset):
    def __getitem__(self, index: int) -> Tuple[list, list, int]:
        l1, l2 = self.data[index]
        label = self._get_label(index)
        l1 = [self.dataset[x][0] for x in l1]
        l2 = [self.dataset[x][0] for x in l2]
        if self.only_one_x_value:
            return [torch.cat((l1[0][0], l2[0][0]), 0)], label
        else:
            return l1, l2, label

    def __init__(
        self,
        dataset_name: str,
        function_name: str,
        operator: Callable[[List[int]], int],
        size=1,
        arity=2,
        seed=None,
        only_one_x_value=False,
    ):
        """Generic dataset for operator(img, img) style datasets.

        :param dataset_name: Dataset to use (train, val, test)
        :param function_name: Name of Problog function to query.
        :param operator: Operator to generate correct examples
        :param size: Size of numbers (number of digits)
        :param arity: Number of arguments for the operator
        :param seed: Seed for RNG
        """
        super(GeneralMNISTOperator, self).__init__()
        assert size >= 1
        assert arity >= 1
        self.dataset_name = dataset_name
        self.dataset = datasets[self.dataset_name]
        self.function_name = function_name
        self.operator = operator
        self.size = size
        self.arity = arity
        self.seed = seed
        self.only_one_x_value = only_one_x_value
        mnist_indices = list(range(len(self.dataset)))
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(mnist_indices)
        dataset_iter = iter(mnist_indices)
        # Build list of examples (mnist indices)
        self.data = []
        try:
            while dataset_iter:
                self.data.append(
                    [
                        [next(dataset_iter) for _ in range(self.size)]
                        for _ in range(self.arity)
                    ]
                )
        except StopIteration:
            pass

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

    def _get_label(self, i: int):
        mnist_indices = self.data[i]
        # Figure out what the ground truth is, first map each parameter to the value:
        ground_truth = [
            digits_to_number(self.dataset[j][1] for j in i) for i in mnist_indices
        ]
        # Then compute the expected value:
        expected_result = self.operator(ground_truth)
        return expected_result

    def __len__(self):
        return len(self.data)