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
from data.network_torch import Net_CiteSeer, Net_Cora, Net_PubMed

# import the given dataset, shuffle it with the given seed and move the given ratio from the train
# set to the test set 
def import_datasets(dataset, move_to_test_set_ratio, seed):
    DATA_ROOT = Path(__file__).parent.parent.joinpath('data')
    data = torch_geometric.datasets.Planetoid(root=str(DATA_ROOT), name=dataset, split="full")
    citation_graph = data[0]

    # variable that holds the examples from the training set that will be added to the test set
    test_set_to_add = []

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

        # move train examples to the test set according to the given ratio
        if dataset_name == "train":
            if move_to_test_set_ratio > 0:
                split_index = round(move_to_test_set_ratio * len(dataset))
                train_set = dataset[split_index:]
                for elem in dataset[:split_index]:
                    test_set_to_add.append(elem)
            else:
                train_set = dataset
        elif dataset_name == "val":
            val_set = dataset
        elif dataset_name == "test":
            test_set = dataset
            for elem in test_set_to_add:
                test_set.append(elem)

    print("The training set contains", len(train_set), "instances.")
    print("The validation set contains", len(val_set), "instances.")
    print("The testing set contains", len(test_set), "instances.")

    return train_set, val_set, test_set

# train the model
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

# test the model
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

################################################# DATASET ###############################################
dataset = "PubMed"
move_to_test_set_ratio = 0
#########################################################################################################

############################################### PARAMETERS ##############################################
seed = 0
nb_epochs = 100
batch_size = 16
learning_rate = 0.001
dropout_rate = 0
#########################################################################################################

# setting seeds for reproducibility
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)

# import train and val set
train_set, val_set, _ = import_datasets(dataset, move_to_test_set_ratio, seed)
train_dataloader = DataLoader(train_set, batch_size=batch_size)
val_dataloader = DataLoader(val_set, batch_size=1)

# create model
if dataset == "CiteSeer":
    model = Net_CiteSeer(dropout_rate)
elif dataset == "Cora":
    model = Net_Cora(dropout_rate)
elif dataset == "PubMed":
    model = Net_PubMed(dropout_rate)

# create loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_accuracy = 0

# training and testing
for epoch in range(nb_epochs):
    # training
    train(train_dataloader, model, loss_fn, optimizer)

    # generate name of file that holds the trained model
    model_file_name = "NN_param_{}_{}_{}_{}_{}".format(seed, epoch + 1, batch_size, learning_rate, 
        dropout_rate)

    # save trained model to a file
    with open(f'results/{dataset}/param/{model_file_name}', "wb") as handle:
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
        "dropout_rate": dropout_rate,
        "accuracy": accuracy,
        "model_file": model_file_name
    }
    with open(f'results/{dataset}/summary_param.json', "a") as outfile:
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