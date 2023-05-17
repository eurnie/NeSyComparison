import random
import torch
import torch_geometric
from pathlib import Path
from deepstochlog.term import Term, List
from deepstochlog.context import ContextualizedTerm, Context
from deepstochlog.dataset import ContextualizedTermDataset

def create_indices(mask):
    indices = []
    for i in range(len(mask)):
        if mask[i]:
            indices.append(i)
    return indices

def import_data(dataset, to_unsupervised, seed):
    DATA_ROOT = Path(__file__).parent.parent.joinpath('data')
    dataset = torch_geometric.datasets.Planetoid(root=str(DATA_ROOT), name=dataset, split="full")
    citation_graph = dataset[0]
    documents = citation_graph.x
    labels = citation_graph.y.numpy()
    
    train_ids = create_indices(citation_graph.train_mask)
    val_ids = create_indices(citation_graph.val_mask)
    test_ids = create_indices(citation_graph.test_mask)

    # move train examples to the unsupervised setting according to the given ratio
    if to_unsupervised > 0:
        split_index = round(to_unsupervised * len(train_ids))
        train_ids = train_ids[split_index:] 

    # shuffle train set
    rng = random.Random(seed)
    rng.shuffle(train_ids)

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

    print("The training set contains", len(train_ids), "instances.")
    print("The validation set contains", len(val_ids), "instances.")
    print("The testing set contains", len(test_ids), "instances.")

    return train_ids, val_ids, test_ids, documents, labels, citations

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
        self, split, ids, labels, documents):
        if split == "train" or split =="valid" or split == "test":
            self.ids = ids
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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        if type(item) is slice:
            return (self[i] for i in range(*item.indices(len(self))))
        return self.dataset[item]
    
def get_datasets(dataset, move_to_test_set_ratio, seed):
    train_ids, val_ids, test_ids, documents, labels, citations = import_data(dataset, move_to_test_set_ratio, seed)

    train_dataset = CiteseerDataset(split="train", ids=train_ids, documents=documents, labels=labels)
    valid_dataset = CiteseerDataset(split="valid", ids=val_ids, documents=documents, labels=labels)
    test_dataset = CiteseerDataset(split="test", ids=test_ids, documents=documents, labels=labels)

    queries_for_model = train_dataset.queries_for_model + valid_dataset.queries_for_model + test_dataset.queries_for_model

    return train_dataset, valid_dataset, test_dataset, queries_for_model, citations