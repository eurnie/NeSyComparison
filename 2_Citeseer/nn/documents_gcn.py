import sys
import random
import numpy
import time
import torch
import torch.nn.functional as F
import torch_geometric
from pathlib import Path

sys.path.append("..")
from data.network_torch import Net_NN

# https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html

def train_and_test(dataset, nb_epochs, learning_rate):
    model = Net_NN()
    data = dataset[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # training
    start_time = time.time()
    for _ in range(0, nb_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    training_time = time.time() - start_time

    # testing
    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    accuracy = int(correct) / int(data.test_mask.sum())

    return accuracy, training_time

############################################### PARAMETERS ##############################################
nb_epochs = 100
learning_rate = 0.001
# weight_decay=5e-4
#########################################################################################################

for seed in range(0, 10):
    # setting seeds for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    # import train and test set
    DATA_ROOT = Path(__file__).parent.parent.joinpath('data')
    dataset = torch_geometric.datasets.Planetoid(root=str(DATA_ROOT), name="CiteSeer", split="full")

    # print("--- Summary of dataset ---")
    # print("Dataset name:", dataset)
    # print("Length of the dataset:", len(dataset))
    # print("Number of classes:", dataset.num_classes)
    # print("Number of features for each node:", dataset.num_node_features)
    # graph = dataset[0]
    # print("Summary of the graph:", graph)
    # print("Undirected graph:", graph.is_undirected())
    # nb_training_entries = graph.train_mask.sum().item()
    # nb_validation_entries = graph.val_mask.sum().item()
    # nb_testing_entries = graph.test_mask.sum().item()
    # print(nb_training_entries, "training entries")
    # print(nb_validation_entries, "validation entries")
    # print(nb_testing_entries, "testing entries")

    # write links to file
    # list_1 = dataset[0].edge_index[0]
    # list_2 = dataset[0].edge_index[1]
    # with open("test.txt", "a") as f:
    #     for i in range(0, len(list_1)):
    #             f.write("linked({},{}).".format(list_1[i], list_2[i]))
    #             f.write("\n")

    # for i in range(len(dataset[0].x)):
    #     with open("test.txt", "a") as f:
    #             f.write("data({},{}).".format(dataset[0].x[i], i))
    #             f.write("\n")

    # train and test the method on the MNIST addition dataset
    accuracy, training_time = train_and_test(dataset, nb_epochs, learning_rate)

    # print results
    print("############################################")
    print("Seed: {} \nAccuracy: {} \nTraining time: {}".format(seed, accuracy, training_time))
    print("############################################")