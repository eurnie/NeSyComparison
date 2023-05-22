import random
import torch_geometric
from pathlib import Path

HIGHEST_ALLOWED_INDEX = 125

def write_to_file(dataset_name, dataset, filename):   
    with open(filename, 'w+') as f:
        for ind, doc, label in dataset:
            if ind < HIGHEST_ALLOWED_INDEX:
                string_to_write = ''
                if dataset_name == 'train':
                    string_to_write += 'document_label(doc_'
                    string_to_write += str(ind)
                    string_to_write += ','
                    string_to_write += str(label.item())
                    string_to_write += ').'
                else:
                    string_to_write += 'unknown(doc_'
                    string_to_write += str(ind)
                    string_to_write += ').'
                f.write(string_to_write)
                f.write('\n')

def write_to_file_txt(dataset, filename):   
    with open(filename, 'w+') as f:
        for ind, doc, label in dataset:
            if ind < HIGHEST_ALLOWED_INDEX:
                string_to_write = 'doc_'
                string_to_write += str(ind)
                string_to_write += ' '
                string_to_write += str(label.item())
                f.write(string_to_write)
                f.write('\n')

def create_datasets(train_set, val_set, test_set):
    for dataset_name in ["train", "val", "test"]:
        if dataset_name == "train":
            data = train_set
        elif dataset_name == "val":
            data = val_set
        elif dataset_name == "test":
            data = test_set

        if dataset_name == 'train':
            write_to_file(dataset_name, data, "data/{}.nl".format(dataset_name))
        else:
            write_to_file(dataset_name, data, "data/{}.nl".format(dataset_name))
            write_to_file_txt(data, "data/{}.txt".format(dataset_name))

    filenames = ['rules.nl', 'data/cites.nl', 'data/train.nl', 'data/val.nl', 'data/test.nl']
    with open('data/citeseer.nl', 'w+') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
                outfile.write('\n')

def import_datasets(dataset, to_unsupervised, seed):
    DATA_ROOT = Path(__file__).parent.parent.joinpath('data')
    data = torch_geometric.datasets.Planetoid(root=str(DATA_ROOT), name=dataset, split="full")
    citation_graph = data[0]

    # create the train, val and test set
    for dataset_name in ["train", "val", "test"]:
        if (dataset_name == "train"):
            mask = citation_graph.train_mask
        elif (dataset_name == "val"):
            mask = citation_graph.val_mask
        elif (dataset_name == "test"):
            mask = citation_graph.test_mask

        indices = []
        for i, bool in enumerate(mask):
            if bool:
                indices.append(i)

        x = citation_graph.x[mask]
        y = citation_graph.y[mask]

        # shuffle dataset
        dataset = [(indices[i], x[i], y[i]) for i in range(len(x))]
        rng = random.Random(seed)
        rng.shuffle(dataset)

        # move train examples to the unsupervised setting according to the given ratio
        if dataset_name == "train":
            if to_unsupervised > 0:
                split_index = round(to_unsupervised * len(dataset))
                train_set = dataset[split_index:]
            else:
                train_set = dataset
        elif dataset_name == "val":
            val_set = dataset
        elif dataset_name == "test":
            test_set = dataset

    print("The training set contains", len(train_set), "instances.")
    print("The validation set contains", len(val_set), "instances.")
    print("The testing set contains", len(test_set), "instances.")

    # write cites to file
    list_1 = citation_graph.edge_index[0]
    list_2 = citation_graph.edge_index[1]
    with open("data/cites.nl", 'w+') as f:
        for i in range(0, len(list_1)):
            if (list_1[i] < HIGHEST_ALLOWED_INDEX) and (list_2[i] < HIGHEST_ALLOWED_INDEX):
                f.write("cite(doc_{},doc_{}).".format(list_1[i], list_2[i]))
                f.write("\n")

    return train_set, val_set, test_set

def generate_dataset(dataset, to_unsupervised, seed):
    assert dataset == "CiteSeer"

    if dataset == "CiteSeer":
        train_set, val_set, test_set = import_datasets(dataset, to_unsupervised, seed)
        create_datasets(train_set, val_set, test_set)
