import sys
import os
import json
import random
import numpy
import time
import torch
import pickle
import torch.nn.functional as F
from torch import nn
import torch_geometric
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append("..")
from data.network_torch import Net, Net_Dropout

# https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html

def import_data(dataset):
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

    dataset_return = [(x[i], y[i]) for i in range(len(x))]
    print("The", print_string, "set contains", len(dataset_return), "instances.")
    return dataset_return

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
    # list_1 = citation_graph.edge_index[0]
    # list_2 = citation_graph.edge_index[1]
    # with open("test.txt", "a") as f:
    #     for i in range(0, len(list_1)):
    #             f.write("cite({},{}).".format(list_1[i], list_2[i]))
    #             f.write("\n")

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for (x, y) in dataloader:
        # compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(dataloader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total += len(x)
    return correct / total

def train_and_test(model_file_name, train_set, val_set, test_set, nb_epochs, batch_size, learning_rate, 
                   use_dropout):
    if use_dropout:
        model = Net_Dropout()
    else:
        model = Net()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataloader = DataLoader(train_set, batch_size=batch_size)
    val_dataloader = DataLoader(val_set, batch_size=1)
    test_dataloader = DataLoader(test_set, batch_size=1)

    # training (with early stopping)
    total_training_time = 0
    best_accuracy = 0
    counter = 0
    for epoch in range(nb_epochs):
        start_time = time.time()
        train(train_dataloader, model, loss_fn, optimizer)
        total_training_time += time.time() - start_time
        val_accuracy = test(val_dataloader, model)
        print("Val accuracy after epoch", epoch, ":", val_accuracy)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            with open("best_model.pickle", "wb") as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            counter = 0
        else:
            if counter >= 2:
                break
            counter += 1
    with open("best_model.pickle", "rb") as handle:
        model = pickle.load(handle)

    os.remove("best_model.pickle")

    # save trained model to a file
    with open("results/final/{}".format(model_file_name), "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    # testing
    start_time = time.time()
    accuracy = test(test_dataloader, model)
    testing_time = time.time() - start_time

    return accuracy, total_training_time, testing_time    
    
############################################### PARAMETERS ##############################################
nb_epochs = 10
batch_size = 64
learning_rate = 0.001
use_dropout = True
#########################################################################################################

for seed in range(0, 10):
    # setting seeds for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    # import train, val and test set
    train_set = import_data("train")
    val_set = import_data("val")
    test_set = import_data("test")

    # generate name of file that holds the trained model
    model_file_name = "NN_final_{}_{}_{}_{}_{}".format(seed, nb_epochs, batch_size, learning_rate, 
        use_dropout)

    # train and test
    accuracy, training_time, testing_time = train_and_test(model_file_name, train_set, val_set,
        test_set, nb_epochs, batch_size, learning_rate, use_dropout)
    
    # save results to a summary file
    information = {
        "algorithm": "NN",
        "seed": seed,
        "nb_epochs": nb_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "use_dropout": use_dropout,
        "accuracy": accuracy,
        "training_time": training_time,
        "testing_time": testing_time,
        "model_file": model_file_name
    }
    with open("results/summary_final.json", "a") as outfile:
        json.dump(information, outfile)
        outfile.write('\n')

    # print results
    print("############################################")
    print("Seed: {} \nAccuracy: {} \nTraining time: {} \nTesting time: {}".format(seed, accuracy, 
        training_time, testing_time))
    print("############################################")