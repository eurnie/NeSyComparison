import torch_geometric
import random
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset as TorchDataset
from problog.logic import Term, list2term, Constant
from deepproblog.dataset import Dataset
from deepproblog.query import Query
from deepproblog.dataset import Dataset

DATA_ROOT = Path(__file__).parent.parent.joinpath('data')
data = torch_geometric.datasets.Planetoid(root=str(DATA_ROOT), name="CiteSeer", split="full")
citation_graph = data[0]
x_values = citation_graph.x
y_values = citation_graph.y

def import_indices(dataset, seed):
    if (dataset == "train"):
        criteria = citation_graph.train_mask
    elif (dataset == "val"):
        criteria = citation_graph.val_mask
    elif (dataset == "test"):
        criteria = citation_graph.test_mask

    indices = []
    labels = []
    for i in range(len(x_values)):
        if criteria[i]:
            indices.append(i)
            labels.append(y_values[i])

    temp = list(zip(indices, labels))
    rng = random.Random(seed)
    rng.shuffle(temp)
    indices, labels = zip(*temp)
    indices, labels = list(indices), list(labels)

    return indices, labels

class Citeseer_Documents(object):
    def __init__(self, x):
        self.data = x

    def __getitem__(self, item):
        return x_values[int(item[0])]

def import_datasets(seed):
    train_set = CiteseerOperator(
        dataset_name="train",
        function_name="document_label",
        seed=seed
    )
    val_set = CiteseerOperator(
        dataset_name="val",
        function_name="document_label"
    )
    test_set = CiteseerOperator(
        dataset_name="test",
        function_name="document_label"
    )
    return train_set, val_set, test_set

class CiteseerOperator(Dataset, TorchDataset):
    def __getitem__(self, index: int) -> Tuple[list, list, int]:
        ind = self.data[index]
        x_values[ind]
        label = self._get_label(index)
        return x_values[ind], label

    def __init__(
        self,
        dataset_name: str,
        function_name: str,
        seed=0
    ):
        super(CiteseerOperator, self).__init__()
        self.dataset_name = dataset_name
        self.dataset = x_values
        self.function_name = function_name
        self.data, self.labels = import_indices(self.dataset_name, seed)

    def to_query(self, i: int) -> Query:
        citeseer_ind = self.data[i]
        expected_result = self._get_label(i)

        return Query(
            Term(
                self.function_name,
                Constant(citeseer_ind),
                Constant(expected_result),
            )
        )

    def _get_label(self, i: int):
        return self.labels[i].item()

    def __len__(self):
        return len(self.data)

citeseer_examples = Citeseer_Documents(x_values)