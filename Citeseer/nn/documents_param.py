import random
import numpy
import sys
import os
import json
import torch
import pickle
import torch_geometric
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler

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
    elif (dataset == "val"):
        x = citation_graph.x[citation_graph.val_mask]
        y = citation_graph.y[citation_graph.val_mask]
    elif (dataset == "test"):
        x = citation_graph.x[citation_graph.test_mask]
        y = citation_graph.y[citation_graph.test_mask]

    dataset_return = [(x[i], y[i]) for i in range(len(x))]
    return dataset_return

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

def train_and_test(model_file_name_dir, total_train_set, nb_epochs, batch_size, learning_rate, use_dropout):
    accuracies = []
    kfold = KFold(n_splits=10, shuffle=True)

    for fold_nb, (train_ids, valid_ids) in enumerate(kfold.split(total_train_set)):
        train_subsampler = SubsetRandomSampler(train_ids)
        valid_subsampler = SubsetRandomSampler(valid_ids)

        train_dataloader = DataLoader(total_train_set, batch_size=batch_size, sampler=train_subsampler)
        test_dataloader = DataLoader(total_train_set, batch_size=1, sampler=valid_subsampler)

        if use_dropout:
            model = Net_Dropout()
        else:
            model = Net()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # training
        for epoch in range(nb_epochs):
            train(train_dataloader, model, loss_fn, optimizer)
            print("Epoch", epoch + 1, "finished.")

        # save trained model to a file
        path = "results/param/{}".format(model_file_name_dir)
        if not os.path.exists(path):
            os.makedirs(path)
        with open("results/param/{}/fold_{}".format(model_file_name_dir, fold_nb), "wb+") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # testing
        fold_accuracy = test(test_dataloader, model)
        accuracies.append(fold_accuracy)
        print(fold_nb + 1, "-- Fold accuracy: ", fold_accuracy)

    return accuracies, sum(accuracies) / 10

############################################### PARAMETERS ##############################################
seed = 0
nb_epochs = 10
batch_size = 32
learning_rate = 0.001
use_dropout = True
#########################################################################################################

# setting seeds for reproducibility
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)

# import train and val set
train_set = import_data("train")
val_set = import_data("val")
total_train_set = train_set + val_set
print("The training set contains", len(total_train_set), "instances.")

# generate name of folder that holds all the trained models
model_file_name_dir = "NN_param_{}_{}_{}_{}_{}".format(seed, nb_epochs, batch_size, learning_rate, 
    use_dropout)

# train and test
accuracies, avg_accuracy = train_and_test(model_file_name_dir, total_train_set, nb_epochs, batch_size, 
    learning_rate, use_dropout)

# save results to a summary file
information = {
    "algorithm": "NN",
    "seed": seed,
    "nb_epochs": nb_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "use_dropout": use_dropout,
    "accuracies": accuracies,
    "avg_accuracy": avg_accuracy,
    "model_files_dir": model_file_name_dir
}
with open("results/summary_param.json", "a") as outfile:
    json.dump(information, outfile)
    outfile.write('\n')

# print results
print("############################################")
print("Seed: {} \nAccuracy: {}".format(seed, avg_accuracy))
print("############################################")