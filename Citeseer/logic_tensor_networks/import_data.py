import tensorflow as tf
import torch_geometric
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path

def get_dataset(dataset, batch_size):
    DATA_ROOT = Path(__file__).parent.parent.joinpath('data')
    data = torch_geometric.datasets.Planetoid(root=str(DATA_ROOT), name="CiteSeer", split="full")
    citation_graph = data[0]

    if (dataset == "train"):
        x = citation_graph.x[citation_graph.train_mask]
        y = citation_graph.y[citation_graph.train_mask]
        print_string = "training"
    elif (dataset == "val"):
        x = citation_graph.x[citation_graph.val_mask]
        y = citation_graph.y[citation_graph.val_mask]
        print_string = "validation"
    elif (dataset == "test"):
        x = citation_graph.x[citation_graph.test_mask]
        y = citation_graph.y[citation_graph.test_mask]
        print_string = "testing"

    print("The", print_string, "set contains", len(x), "instances.")
    dataset_return = tf.data.Dataset.from_tensor_slices((x,y)).batch(batch_size)
    return dataset_return

def get_cites():
    DATA_ROOT = Path(__file__).parent.parent.joinpath('data')
    data = torch_geometric.datasets.Planetoid(root=str(DATA_ROOT), name="CiteSeer", split="full")
    citation_graph = data[0]

    list_1_return = []
    list_2_return = []

    list_1 = citation_graph.edge_index[0]
    list_2 = citation_graph.edge_index[1]
    for i in range(0, len(list_1)):
        list_1_return.append(citation_graph.x[list_1[i]])
        list_2_return.append(citation_graph.x[list_2[i]])

    return list_1_return, list_2_return