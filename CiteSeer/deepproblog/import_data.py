import torch_geometric
import random
import numpy as np
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset as TorchDataset
from problog.logic import Term, list2term, Constant
from deepproblog.dataset import Dataset
from deepproblog.query import Query
from deepproblog.dataset import Dataset

DATA_ROOT = Path(__file__).parent.parent.joinpath('data')
data_citeseer = torch_geometric.datasets.Planetoid(root=str(DATA_ROOT), name="CiteSeer", split="full")
citation_graph_citeseer = data_citeseer[0]
x_values_citeseer = citation_graph_citeseer.x
y_values_citeseer = citation_graph_citeseer.y

cite_a_citeseer = citation_graph_citeseer.edge_index[0]
cite_b_citeseer = citation_graph_citeseer.edge_index[1]

docs_too_much_cites_citeseer = []
for i in range(len(x_values_citeseer)):
    if np.count_nonzero(cite_a_citeseer.numpy() == i) > 13:
        docs_too_much_cites_citeseer.append(i)

DATA_ROOT = Path(__file__).parent.parent.joinpath('data')
data_cora = torch_geometric.datasets.Planetoid(root=str(DATA_ROOT), name="Cora", split="full")
citation_graph_cora = data_cora[0]
x_values_cora = citation_graph_cora.x
y_values_cora = citation_graph_cora.y

cite_a_cora = citation_graph_cora.edge_index[0]
cite_b_cora = citation_graph_cora.edge_index[1]

docs_too_much_cites_cora = []
for i in range(len(x_values_cora)):
    if np.count_nonzero(cite_a_cora.numpy() == i) > 13:
        docs_too_much_cites_cora.append(i)

def import_indices(used_dataset, split, move_to_unsupervised, seed):
    if used_dataset == "CiteSeer":
        citation_graph = citation_graph_citeseer
        x_values = x_values_citeseer
        y_values = y_values_citeseer
        docs_too_much_cites = docs_too_much_cites_citeseer
    elif used_dataset == "Cora":
        citation_graph = citation_graph_cora
        x_values = x_values_cora
        y_values = y_values_cora
        docs_too_much_cites = docs_too_much_cites_cora

    if (split == "train"):
        criteria = citation_graph.train_mask
    elif (split == "val"):
        criteria = citation_graph.val_mask
    elif (split == "test"):
        criteria = citation_graph.test_mask

    indices = []
    labels = []
    for i in range(len(x_values)):
        if criteria[i]:
            if (split == 'train') or (not i in docs_too_much_cites):
                indices.append(i)
                labels.append(y_values[i])

    # move train examples to the unsupervised setting according to the given ratio
    if move_to_unsupervised > 0:
        if split == "train":
            split_index = round(move_to_unsupervised * len(labels))
            indices = indices[split_index:]
            labels = labels[split_index:]

    temp = list(zip(indices, labels))
    rng = random.Random(seed)
    rng.shuffle(temp)
    indices, labels = zip(*temp)
    indices, labels = list(indices), list(labels)

    assert len(labels) == len(indices)
    print(f'The {split} set contains', len(labels), "instances.")

    return indices, labels

class Citeseer_Documents(object):
    def __init__(self, x):
        self.data = x

    def __getitem__(self, item):
        return self.data[int(item[0])]
    
# class Citeseer_Cites(object):
#     def __init__(self, cite_a, cite_b):
#         self.cites_a = cite_a
#         self.cites_b = cite_b

#     def __getitem__(self, item):
#         print("----------------")
#         indices = []
#         for i in range(len(self.cites_a)):
#             if self.cites_a[i] == int(item[0]):
#                 indices.append(i)

#         return_values = []
#         for index in indices:
#             return_values.append(self.cites_b[index])

#         print(return_values)

#         return return_values

def import_datasets(dataset, move_to_test_set_ratio, seed):
    train_set = CiteseerOperator(
        dataset_name="train",
        function_name="document_label",
        move_to_test_set_ratio=move_to_test_set_ratio,
        seed=seed,
        dataset=dataset
    )
    val_set = CiteseerOperator(
        dataset_name="val",
        function_name="document_label",
        move_to_test_set_ratio=move_to_test_set_ratio,
        seed=seed,
        dataset=dataset
    )
    test_set = CiteseerOperator(
        dataset_name="test",
        function_name="document_label",
        move_to_test_set_ratio=move_to_test_set_ratio,
        seed=seed,
        dataset=dataset
    )
    return train_set, val_set, test_set

class CiteseerOperator(Dataset, TorchDataset):
    def __getitem__(self, index: int) -> Tuple[list, list, int]:
        if self.used_dataset == "CiteSeer":
            x_values = x_values_citeseer
        elif self.used_dataset == "Cora":
            x_values = x_values_cora
        ind = self.data[index]
        x_values[ind]
        label = self._get_label(index)
        return x_values[ind], label

    def __init__(
        self,
        dataset_name: str,
        function_name: str,
        move_to_test_set_ratio=0,
        seed=0,
        dataset="CiteSeer"
    ):
        super(CiteseerOperator, self).__init__()
        self.used_dataset = dataset
        self.dataset_name = dataset_name
        if self.used_dataset == "CiteSeer":
            self.dataset = x_values_citeseer
        elif self.used_dataset == "Cora":
            self.dataset = x_values_cora
        self.function_name = function_name
        self.move_to_test_set_ratio = move_to_test_set_ratio
        self.seed = seed
        self.data, self.labels = import_indices(self.used_dataset, self.dataset_name, self.move_to_test_set_ratio, self.seed)

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

citeseer_examples = Citeseer_Documents(x_values_citeseer)
cora_examples = Citeseer_Documents(x_values_cora)
# citeseer_cites = Citeseer_Cites(cite_a, cite_b)