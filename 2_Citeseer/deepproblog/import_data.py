import torch_geometric
from pathlib import Path
from deepproblog.dataset import Dataset
from deepproblog.query import Query
from problog.logic import Term, Constant
from deepproblog.dataset import Dataset

DATA_ROOT = Path(__file__).parent.parent.joinpath('data')
dataset = torch_geometric.datasets.Planetoid(root=str(DATA_ROOT), name="CiteSeer", split="full")[0]

class Citeseer_Documents(object):
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, item):
        return dataset.x[int(item[0])]

class Citeseer(Dataset):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.data = []

        for i in range(len(dataset.y)):
            if self.dataset_name == "train":
                if dataset.train_mask[i]:
                    self.data.append(str(i))
            elif self.dataset_name == "val":
                if dataset.val_mask[i]:
                    self.data.append(str(i))
            elif self.dataset_name == "test":
                if dataset.test_mask[i]:
                    self.data.append(str(i))
        
        self.labels = dataset.y

    def __len__(self):
        return len(self.data)

    def to_query(self, i):
        l = Constant(self.labels[i].item())
        return Query(
            Term("document_label", Term("tensor", Term(self.dataset_name, Term("a"))), l),
            substitution={Term("a"): Constant(i)},
        )

def import_dataset(dataset: str):
    return Citeseer(
        dataset_name=dataset
    )

Citeseer_train = Citeseer_Documents("train")
Citeseer_val = Citeseer_Documents("val")
Citeseer_test = Citeseer_Documents("test")