import os
import random
import numpy
import sys
import json
import torch
import pickle
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.append("..")
from data.generate_dataset import generate_dataset_mnist, generate_dataset_fashion_mnist
from data.network_torch import Net_NN

def parse_data(dataset, filename, dataset_name, size_val):
    DATA_ROOT = Path(__file__).parent.parent.joinpath('data')

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    datasets_mnist = {
        "train": torchvision.datasets.MNIST(
            root=str(DATA_ROOT), train=True, download=True, transform=transform
        ),
        "test": torchvision.datasets.MNIST(
            root=str(DATA_ROOT), train=False, download=True, transform=transform
        ),
    }

    datasets_fashion_mnist = {
        "train": torchvision.datasets.FashionMNIST(
            root=str(DATA_ROOT), train=True, download=True, transform=transform
        ),
        "test": torchvision.datasets.FashionMNIST(
            root=str(DATA_ROOT), train=False, download=True, transform=transform
        ),
    }

    if dataset == "MNIST":
        datasets = datasets_mnist
    elif dataset == "FashionMNIST":
        datasets = datasets_fashion_mnist

    split_index = round(size_val * 30000)
    if dataset_name == "train":
        dataset_used = "train"
        start = split_index
        end = 30000
    elif dataset_name == "val":
        dataset_used = "train"
        start = 0
        end = split_index
    elif dataset_name == "test":
        dataset_used = "test"
        start = 0
        end = 5000

    with open(filename + dataset_used + ".txt") as f:
        entries = f.readlines()

    dataset = []

    for i in range(start, end):
        index_digit_1 = int(entries[i].split(" ")[0])
        index_digit_2 = int(entries[i].split(" ")[1])
        sum = int(entries[i].split(" ")[2])
        first = datasets[dataset_used][index_digit_1][0][0]
        second = datasets[dataset_used][index_digit_2][0][0]
        new_tensor = torch.cat((first, second), 0)
        new_tensor = new_tensor[None, :]
        dataset.append((new_tensor, sum))
    
    return dataset

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

################################################# DATASET ###############################################
dataset = "MNIST"
#########################################################################################################

############################################### PARAMETERS ##############################################
seed = 0
nb_epochs = 100
size_val = 0.1
#########################################################################################################

for dropout_rate in [0, 0.2]:
    for optimizer_name in ["Adam", "SGD"]:
        for learning_rate in [0.001, 0.0001]:
            for batch_size in [2, 8, 32, 128]:
                # generate name of file that holds the trained model
                model_file_name = "NN_param_{}_{}_{}_{}_{}_{}_{}".format(seed, 
                    nb_epochs, size_val, dropout_rate, optimizer_name, learning_rate, batch_size)
                model_file_location = f'results/{dataset}/param/{model_file_name}'
                
                if not os.path.isfile(model_file_location):
                    # setting seeds for reproducibility
                    random.seed(seed)
                    numpy.random.seed(seed)
                    torch.manual_seed(seed)

                    # generate and shuffle dataset
                    if dataset == "MNIST":
                        generate_dataset_mnist(seed, 0)
                    elif dataset == "FashionMNIST":
                        generate_dataset_fashion_mnist(seed, 0)
                    processed_data_path = f'../data/{dataset}/processed/'

                    # import train, val and test set
                    train_set = parse_data(dataset, processed_data_path, "train", size_val)
                    val_set = parse_data(dataset, processed_data_path, "val", size_val)

                    # create model and optimizer
                    model = Net_NN(dropout_rate)
                    loss_fn = nn.CrossEntropyLoss()
                    if optimizer_name == "Adam":
                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                    elif optimizer_name == "SGD":
                        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

                    train_dataloader = DataLoader(train_set, batch_size=batch_size)
                    val_dataloader = DataLoader(val_set, batch_size=1)

                    # training (with early stopping)
                    best_accuracy = -1
                    counter = 0
                    for epoch in range(nb_epochs):
                        train(train_dataloader, model, loss_fn, optimizer)
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
                        "algorithm": "NN",
                        "seed": seed,
                        "nb_epochs": nb_epochs_done,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "optimizer": optimizer_name,
                        "dropout_rate": dropout_rate,
                        "size_val": size_val,
                        "accuracy": best_accuracy,
                        "model_file": model_file_name
                    }
                    with open(f'results/{dataset}/summary_param.json', "a") as outfile:
                        json.dump(information, outfile)
                        outfile.write('\n')