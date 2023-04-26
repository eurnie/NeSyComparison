import sys
import json
import random
import numpy
import torch
import pickle
from torch import nn
import torch_geometric
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append("..")
from data.network_torch import Net, Net_Dropout

def import_data(dataset_name, seed):
    DATA_ROOT = Path(__file__).parent.parent.joinpath('data')
    data = torch_geometric.datasets.Planetoid(root=str(DATA_ROOT), name="CiteSeer", split="full")
    citation_graph = data[0]

    if (dataset_name == "train"):
        mask = citation_graph.train_mask
        print_string = "training"
    elif (dataset_name == "val"):
        mask = citation_graph.val_mask
        print_string = "validation"
    elif (dataset_name == "test"):
        mask = citation_graph.test_mask
        print_string = "testing"

    indices = []
    for i, bool in enumerate(mask):
        if bool:
            indices.append(i)

    x = citation_graph.x[mask]
    y = citation_graph.y[mask]

    # generate and shuffle dataset
    dataset = [(indices[i], x[i], y[i]) for i in range(len(x))]
    rng = random.Random(seed)
    rng.shuffle(dataset)

    print("The", print_string, "set contains", len(dataset), "instances.")
    return dataset

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for (_, x, y) in dataloader:
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
        for (_, x, y) in dataloader:
            pred = model(x)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total += len(x)
    return correct / total

############################################### PARAMETERS ##############################################
seed = 0
nb_epochs = 100
batch_size = 2
learning_rate = 0.01
use_dropout = False
#########################################################################################################

# setting seeds for reproducibility
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)

# import train and val set
train_set = import_data("train", seed)
val_set = import_data("val", seed)

# create model and loss function
if use_dropout:
    model = Net_Dropout()
else:
    model = Net()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_dataloader = DataLoader(train_set, batch_size=batch_size)
val_dataloader = DataLoader(val_set, batch_size=1)

best_accuracy = 0

# training
for epoch in range(nb_epochs):
    train(train_dataloader, model, loss_fn, optimizer)

    # generate name of file that holds the trained model
    model_file_name = "NN_param_{}_{}_{}_{}_{}".format(seed, epoch + 1, batch_size, learning_rate, 
        use_dropout)

    # save trained model to a file
    with open("results/param/{}".format(model_file_name), "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    # testing
    accuracy = test(val_dataloader, model)

    # save results to a summary file
    information = {
        "algorithm": "NN",
        "seed": seed,
        "nb_epochs": epoch + 1,
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

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        counter = 0
    else:
        if counter >= 2:
            break
        counter += 1