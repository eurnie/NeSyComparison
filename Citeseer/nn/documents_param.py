import sys
import json
import random
import numpy
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

def train_and_test(model_file_name, train_set, val_set, nb_epochs, batch_size, learning_rate, 
                   use_dropout):
    if use_dropout:
        model = Net_Dropout()
    else:
        model = Net()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataloader = DataLoader(train_set, batch_size=batch_size)
    val_dataloader = DataLoader(val_set, batch_size=1)

    # training
    for _ in range(nb_epochs):
        train(train_dataloader, model, loss_fn, optimizer)

    # save trained model to a file
    with open("results/param/{}".format(model_file_name), "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    # testing
    accuracy = test(val_dataloader, model)

    return accuracy  
    
############################################### PARAMETERS ##############################################
seed = 0
nb_epochs = 10
batch_size = 64
learning_rate = 0.001
use_dropout = False
#########################################################################################################

# (2, 8, 0.01, False)
# (3, 16, 0.01, False)
# (1, 2, 0.01, False)
# (2, 2, 0.01, False)
# (2, 16, 0.01, False)
# (1, 16, 0.01, False)
# (1, 8, 0.01, False)
# (3, 8, 0.01, False)
# (2, 4, 0.01, False)
# (3, 2, 0.01, False)
# (3, 4, 0.01, False)
# (1, 4, 0.01, False)

# setting seeds for reproducibility
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)

# import train and val set
train_set = import_data("train")
val_set = import_data("val")

# generate name of file that holds the trained model
model_file_name = "NN_param_{}_{}_{}_{}_{}".format(seed, nb_epochs, batch_size, learning_rate, 
    use_dropout)

# train and test
accuracy = train_and_test(model_file_name, train_set, val_set,
    nb_epochs, batch_size, learning_rate, use_dropout)

# save results to a summary file
information = {
    "algorithm": "NN",
    "seed": seed,
    "nb_epochs": nb_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "use_dropout": use_dropout,
    "accuracy": accuracy,
    "model_file": model_file_name
}
with open("results/summary_param.json", "a") as outfile:
    json.dump(information, outfile)
    outfile.write('\n')

# print results
print("############################################")
print("Seed: {} \nAccuracy: {}".format(seed, accuracy))
print("############################################")