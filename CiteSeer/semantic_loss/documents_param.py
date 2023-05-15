import os
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
from data.network_torch import Net_CiteSeer, Net_Cora, Net_PubMed

# import the dataset, shuffle it with the seed and move examples to the unsupervised setting
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

    # collect the cites
    cites = []
    cites_a = citation_graph.edge_index[0]
    cites_b = citation_graph.edge_index[1]

    for i in range(len(cites_a)):
        cites.append((cites_a[i], cites_b[i]))

    # list that holds the features of all the documents
    ind_to_features = []
    for i in range(len(citation_graph.x)):
        ind_to_features.append(citation_graph.x[i][None,:])

    print("The training set contains", len(train_set), "instances.")
    print("The validation set contains", len(val_set), "instances.")
    print("The testing set contains", len(test_set), "instances.")

    return train_set, val_set, test_set, cites, ind_to_features

# train the model
def train(dataloader, model, cites, ind_to_features, sl, loss_fn, optimizer):
    model.train()
    for (indices, x, y) in dataloader:
        loss = 0
        pred = model(x)

        # loop over all examples in this batch
        for i, ind in enumerate(indices):
            # search for all cites
            for (a, b) in cites:
                if ind == a:
                    # predict the label of the cited example and add semantic loss
                    loss += sl(model(ind_to_features[b])-pred[i])

        # compute prediction error
        loss += loss_fn(pred, y) 

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
dataset = "CiteSeer"
#########################################################################################################

############################################### PARAMETERS ##############################################
seed = 0
nb_epochs = 100
#########################################################################################################

assert (dataset == "CiteSeer") or (dataset == "Cora") or (dataset == "PubMed")

for dropout_rate in [0, 0.2]:
    for optimizer_name in ["Adam", "SGD"]:
        for learning_rate in [0.001, 0.0001]:
            for batch_size in [2, 8, 32, 128]:
                # generate name of file that holds the trained model
                model_file_name = "SL_param_{}_{}_{}_{}_{}_{}".format(seed, 
                    nb_epochs, dropout_rate, optimizer_name, learning_rate, batch_size)
                model_file_location = f'results/{dataset}/param/{model_file_name}'
                
                if not os.path.isfile(model_file_location):
                    # setting seeds for reproducibility
                    random.seed(seed)
                    numpy.random.seed(seed)
                    torch.manual_seed(seed)

                    # import train and val set
                    train_set, val_set, _, cites, ind_to_features = import_datasets(dataset, 0, seed)
                    train_dataloader = DataLoader(train_set, batch_size=batch_size)
                    val_dataloader = DataLoader(val_set, batch_size=1)

                    # create model
                    if dataset == "CiteSeer":
                        model = Net_CiteSeer(dropout_rate)
                    elif dataset == "Cora":
                        model = Net_Cora(dropout_rate)
                    elif dataset == "PubMed":
                        model = Net_PubMed(dropout_rate)

                    # create loss functions
                    sl = SemanticLoss(f'constraints/{dataset}/constraint.sdd', f'constraints/{dataset}/constraint.vtree')
                    loss_fn = nn.CrossEntropyLoss()

                    if optimizer_name == "Adam":
                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                    elif optimizer_name == "SGD":
                        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

                    # training (with early stopping)
                    best_accuracy = -1
                    counter = 0
                    for epoch in range(nb_epochs):
                        train(train_dataloader, model, cites, ind_to_features, sl, loss_fn, optimizer)
                        val_accuracy = test(val_dataloader, model)
                        print("Val accuracy after epoch", epoch, ":", val_accuracy)
                        if val_accuracy > best_accuracy:
                            best_accuracy = val_accuracy
                            nb_epochs_done = epoch + 1
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
                    with open(model_file_location, "wb") as handle:
                        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        
                    # save results to a summary file
                    information = {
                        "algorithm": "SL",
                        "seed": seed,
                        "nb_epochs": nb_epochs_done,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "optimizer": optimizer_name,
                        "dropout_rate": dropout_rate,
                        "accuracy": best_accuracy,
                        "model_file": model_file_name
                    }
                    with open(f'results/{dataset}/summary_param.json', "a") as outfile:
                        json.dump(information, outfile)
                        outfile.write('\n')