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
import torch_geometric

DATA_ROOT = Path(__file__).parent.parent.joinpath('data')
data = torch_geometric.datasets.Planetoid(root=str(DATA_ROOT), name="CiteSeer", split="full")
citation_graph = data[0]
x_values = citation_graph.x
y_values = citation_graph.y

def import_indices(dataset):
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

    return indices, labels

class Citeseer_Documents(object):
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, item):
        return x_values[int(item[0])]

def import_datasets():
    train_set = CiteseerOperator(
        dataset_name="train",
        function_name="document_label"
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
        function_name: str
    ):
        super(CiteseerOperator, self).__init__()
        self.dataset_name = dataset_name
        self.dataset = x_values
        self.function_name = function_name
        self.data, self.labels = import_indices(self.dataset_name)

    def to_query(self, i: int) -> Query:
        """Generate queries"""
        citeseer_ind = self.data[i]
        expected_result = self._get_label(i)

        # Build substitution dictionary for the arguments
        subs = dict()
        # var_names = []
        # for i in range(self.arity):
        #     inner_vars = []
        #     for j in range(self.size):
        #         t = Term(f"p{i}_{j}")
        #         subs[t] = Term(
        #             "tensor",
        #             Term(
        #                 self.dataset_name,
        #                 Constant(mnist_indices[i][j]),
        #             ),
        #         )
        #         inner_vars.append(t)
        #     var_names.append(inner_vars)

       # t = Term(f"p{i}_")
       ## subs[t] = Term(
       #     "tensor",
       #     Term(
       #         self.dataset_name,
       #         Constant(citeseer_ind),
       #     ),
       # )

        return Query(
            Term(
                self.function_name,
                Constant(citeseer_ind),
                Constant(expected_result),
            )
        )

        # # Build query
        # if self.size == 1:
        #     return Query(
        #         Term(
        #             self.function_name,
        #             *(e[0] for e in var_names),
        #             Constant(expected_result),
        #         ),
        #         subs,
        #     )
        # else:
        #     return Query(
        #         Term(
        #             self.function_name,
        #             *(list2term(e) for e in var_names),
        #             Constant(expected_result),
        #         ),
        #         subs,
        #     )

    def _get_label(self, i: int):
        return self.labels[i].item()

    def __len__(self):
        return len(self.data)

Citeseer_train = Citeseer_Documents("train")
#Citeseer_val = Citeseer_Documents("val")
#Citeseer_test = Citeseer_Documents("test")