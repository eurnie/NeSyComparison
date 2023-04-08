import random
import time
import sys
import os
import json
import torch
import pickle
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from copy import deepcopy

sys.path.append("..")
from data.generate_dataset import generate_dataset
from data.network_torch import Net_NN, Net_NN_Dropout, Net_NN_Extra

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
        model = Net_NN_Dropout()
    else:
        model = Net_NN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    training_data = []

    for map_id in range(len(train_set["maps"])):
        map = train_set["maps"][map_id]
        cost = train_set["costs"][map_id]
        for target_id in range(len(train_set["targets"][map_id])):
            target = train_set["targets"][map_id][target_id]
            for source_id in range(len(train_set["sources"][map_id])):
                source = train_set["sources"][map_id][target_id][source_id]
                path = train_set["paths"][map_id][target_id][source_id]

                x_value = cost.flatten()
                np.append(x_value, source)
                np.append(x_value, target)

                y_value = path.flatten()

                print(x_value)

                training_data.append((x_value, y_value))



                # display_map_and_path(map, path, source, target)



    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    # val_dataloader = DataLoader(val_set, batch_size=1)
    # test_dataloader = DataLoader(test_set, batch_size=1)



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
nb_epochs = 100
batch_size = 8
learning_rate = 0.001
use_dropout = False
size_val = 0.1
#########################################################################################################

for seed in range(0, 10):
    # setting seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # shuffle dataset
    # TODO

    # import train, val and test set
    train_set = load_npz("../data/train.npz")
    val_set = load_npz("../data/val.npz")
    test_set = load_npz("../data/test.npz")

    # generate name of file that holds the trained model
    model_file_name = "NN_final_{}_{}_{}_{}_{}_{}".format(seed, nb_epochs, batch_size, learning_rate, 
        use_dropout, size_val)

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
        "size_val": size_val,
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