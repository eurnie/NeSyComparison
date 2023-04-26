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
from semantic_loss_pytorch import SemanticLoss

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

def import_cites():
    DATA_ROOT = Path(__file__).parent.parent.joinpath('data')
    data = torch_geometric.datasets.Planetoid(root=str(DATA_ROOT), name="CiteSeer", split="full")
    citation_graph = data[0]

    cites = []
    cites_a = citation_graph.edge_index[0]
    cites_b = citation_graph.edge_index[1]

    for i in range(len(cites_a)):
        cites.append((cites_a[i], cites_b[i]))

    ind_to_features = []
    for i in range(len(citation_graph.x)):
        if citation_graph.train_mask[i]:
            ind_to_features.append(citation_graph.x[i][None,:])
        else:
            ind_to_features.append(None)

    return cites, ind_to_features

def train(dataloader, ind_to_features, cites, model, sl, sl_one_hot_encoding, loss_fn, optimizer):
    model.train()
    for (indices, x, y) in dataloader:
        loss = 0
        pred = model(x)

        # loop over all examples in this batch
        for i, ind in enumerate(indices):
            # search for all cites
            for (a, b) in cites:
                if ind == a:
                    # check if the cited example is a train example
                    if ind_to_features[b] is not None:
                        # predict the label of the cited train example and add extra loss
                        loss += sl(model(ind_to_features[b])-pred[i])

        # one-hot-encoding constraint loss
        if sl_one_hot_encoding is not None:
            loss += sl_one_hot_encoding(pred)

        # compute prediction error
        loss += loss_fn(pred, y) 

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
batch_size = 4
learning_rate = 0.001
use_dropout = False
use_one_hot_encoding_constraint = True
#########################################################################################################

# setting seeds for reproducibility
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)

# import train and val set
train_set = import_data("train", seed)
val_set = import_data("val", seed)
cites, ind_to_features = import_cites()

# create model and loss functions
if use_dropout:
    model = Net_Dropout()
else:
    model = Net()
sl = SemanticLoss('constraint.sdd', 'constraint.vtree')
if use_one_hot_encoding_constraint:
    sl_one_hot_encoding = SemanticLoss('constraint_one_hot_encoding.sdd', 'constraint_one_hot_encoding.vtree')
else:
    sl_one_hot_encoding = None
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# create dataloaders
train_dataloader = DataLoader(train_set, batch_size=batch_size)
val_dataloader = DataLoader(val_set, batch_size=1)

best_accuracy = 0

# training and testing on val set
for epoch in range(nb_epochs):
    # train model for an extra epoch
    train(train_dataloader, ind_to_features, cites, model, sl, sl_one_hot_encoding, loss_fn, optimizer)

    # generate name of file that holds the trained model
    model_file_name = "SL_param_{}_{}_{}_{}_{}_{}".format(seed, epoch + 1, batch_size, learning_rate, 
        use_dropout, use_one_hot_encoding_constraint)

    # save trained model to a file
    with open("results/param/{}".format(model_file_name), "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    # testing on val set
    accuracy = test(val_dataloader, model)

    # save results to a summary file
    information = {
        "algorithm": "SL",
        "seed": seed,
        "nb_epochs": epoch + 1,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "use_dropout": use_dropout,
        "use_one_hot_encoding_constraint": use_one_hot_encoding_constraint,
        "accuracy": accuracy,
        "model_file": model_file_name
    }
    with open("results/summary_param.json", "a") as outfile:
        json.dump(information, outfile)
        outfile.write('\n')

    # print results
    print("############################################")
    print("Accuracy: {}".format(accuracy))
    print("############################################")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        counter = 0
    else:
        if counter >= 2:
            break
        counter += 1