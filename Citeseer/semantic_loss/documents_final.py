import sys
import os
import json
import random
import numpy
import time
import torch
import pickle
from torch import nn
import torch_geometric
from pathlib import Path
from torch.utils.data import DataLoader
from semantic_loss_pytorch import SemanticLoss

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
    test_set_to_add_ind = []

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
                    test_set_to_add_ind.append(elem[0])
            else:
                train_set = dataset
        elif dataset_name == "val":
            val_set = dataset
        elif dataset_name == "test":
            test_set = dataset
            for elem in test_set_to_add:
                test_set.append(elem)

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

def train_and_test(dataset, model_file_name, train_set, val_set, test_set, cites, ind_to_features, 
                   nb_epochs, batch_size, learning_rate, dropout_rate):
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
        train(train_dataloader, model, cites, ind_to_features, sl, loss_fn, optimizer)
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
    with open(f'results/{dataset}/final/{model_file_name}', "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    # testing
    start_time = time.time()
    accuracy = test(test_dataloader, model)
    testing_time = time.time() - start_time

    return accuracy, total_training_time, testing_time    
    
################################################# DATASET ###############################################
dataset = "CiteSeer"
move_to_test_set_ratio = 0
#########################################################################################################

############################################### PARAMETERS ##############################################
nb_epochs = 100
batch_size = 8
learning_rate = 0.001
dropout_rate = 0
#########################################################################################################

assert (dataset == "CiteSeer") or (dataset == "Cora") or (dataset == "PubMed")

for seed in range(0, 10):
    # setting seeds for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    # import train, val and test set
    train_set, val_set, test_set, cites, ind_to_features = import_datasets(dataset, move_to_test_set_ratio, 
                                                                           seed)

    # generate name of file that holds the trained model
    model_file_name = "SL_final_{}_{}_{}_{}_{}".format(seed, nb_epochs, batch_size, learning_rate, 
        dropout_rate)

    # train and test
    accuracy, training_time, testing_time = train_and_test(dataset, model_file_name, train_set, val_set,
        test_set, cites, ind_to_features, nb_epochs, batch_size, learning_rate, dropout_rate)
    
    # save results to a summary file
    information = {
        "algorithm": "SL",
        "seed": seed,
        "nb_epochs": nb_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dropout_rate": dropout_rate,
        "accuracy": accuracy,
        "training_time": training_time,
        "testing_time": testing_time,
        "model_file": model_file_name
    }
    with open(f'results/{dataset}/summary_final.json', "a") as outfile:
        json.dump(information, outfile)
        outfile.write('\n')

    # print results
    print("############################################")
    print("Seed: {} \nAccuracy: {} \nTraining time: {} \nTesting time: {}".format(seed, accuracy, 
        training_time, testing_time))
    print("############################################")