import dgl
import random
import torch
import torch_geometric
from pathlib import Path
from deepstochlog.term import Term, List
from deepstochlog.context import ContextualizedTerm, Context
from deepstochlog.dataset import ContextualizedTermDataset

DATA_ROOT = Path(__file__).parent.parent.joinpath('data')
dataset = torch_geometric.datasets.Planetoid(root=str(DATA_ROOT), name="CiteSeer", split="full")
citation_graph = dataset[0]
documents = citation_graph.x
labels = citation_graph.y.numpy()

def import_indices(dataset):
    if (dataset == "train"):
        criteria = citation_graph.train_mask
    elif (dataset == "val"):
        criteria = citation_graph.val_mask
    elif (dataset == "test"):
        criteria = citation_graph.test_mask

    indices = []
    for i in range(len(documents)):
        if criteria[i]:
            indices.append(i)

    return indices

train_ids = import_indices("train")
val_ids = import_indices("val")
test_ids = import_indices("test")

edges = []
citations = []

list_1 = citation_graph.edge_index[0]
list_2 = citation_graph.edge_index[1]
for eid in range(len(list_1)):
    a = list_1[eid]
    b = list_2[eid]
    edges.append((a,b))
    citations.append("cite(%d, %d)." % (a,b))
citations = "\n".join(citations)

####################

# root_path = Path(__file__).parent
# dataset = dgl.data.CiteseerGraphDataset()
# g = dataset[0]

# # get node feature
# documents = g.ndata['feat']
# # get data split
# train_ids = np.where(g.ndata['train_mask'].numpy())[0]
# val_ids = np.where(g.ndata['val_mask'].numpy())[0]
# test_ids = np.where(g.ndata['test_mask'].numpy())[0]

# # get labels
# labels = g.ndata['label'].numpy()
# # edges = []
# citations = []
# for eid in range(g.num_edges()):
#     a, b = g.find_edges(eid)
#     a, b = a.numpy().tolist()[0], b.numpy().tolist()[0],
#     edges.append((a,b))
#     citations.append("cite(%d, %d)." % (a,b))
# citations = "\n".join(citations)

class CiteseerDataset(ContextualizedTermDataset):
    def __init__(
        self,
        split: str,
        labels,
        documents,
        seed):
        if split == "train":
            self.ids = train_ids
        elif split =="valid":
            self.ids = val_ids
        elif split == "test":
            self.ids = test_ids
        else:
            raise Exception("Unknown split %s" % split)
        self.labels = labels
        self.is_test = True if split in ("test", "valid") else False
        self.documents = documents
        self.dataset = []

        context = {Term(str(i)): d for i, d in enumerate(self.documents)}
        for i in range(6):
            context[Term(str(i))] = torch.tensor([i])
        context = Context(context)
        self.queries_for_model = []
        for did in self.ids:
            label = List(str(self.labels[did]))
            query = ContextualizedTerm(
                context=context,
                term=Term("s", did, label))
            self.dataset.append(query)
            if self.is_test:
                query_model = Term("s", did, List("_"))
            else:
                query_model = query.term
            self.queries_for_model.append(query_model)

        # shuffle dataset
        if split == "train":
            rng = random.Random(seed)
            rng.shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        if type(item) is slice:
            return (self[i] for i in range(*item.indices(len(self))))
        return self.dataset[item]
    
def get_dataset(dataset_name, seed):
    train_dataset = CiteseerDataset(split="train", documents=documents, labels=labels, seed=seed)
    valid_dataset = CiteseerDataset(split="valid", documents=documents, labels=labels, seed=seed)
    test_dataset = CiteseerDataset(split="test", documents=documents, labels=labels, seed=seed)

    queries_for_model = train_dataset.queries_for_model + valid_dataset.queries_for_model + test_dataset.queries_for_model

    if dataset_name == "train":
        return train_dataset, queries_for_model
    elif dataset_name == "val":
        return valid_dataset, queries_for_model
    elif dataset_name == "test":
        return test_dataset, queries_for_model