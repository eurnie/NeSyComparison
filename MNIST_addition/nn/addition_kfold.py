import random
import numpy
import sys
import os
import json
import torch
import pickle
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler

sys.path.append("..")
from data.generate_dataset import generate_dataset_mnist, generate_dataset_fashion_mnist
from data.network_torch import Net_NN

def parse_data(dataset, filename):
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

    if dataset == "mnist":
        datasets = datasets_mnist
    elif dataset == "fashion_mnist":
        datasets = datasets_fashion_mnist

    dataset_used = "train"

    with open(filename + dataset_used + ".txt") as f:
        entries = f.readlines()

    dataset = []

    for i in range(0, 30000):
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

def train_and_test(dataset, model_file_name_dir, total_train_set, nb_epochs, batch_size, 
                   learning_rate, use_dropout):
    accuracies = []
    kfold = KFold(n_splits=10, shuffle=True)

    for fold_nb, (train_ids, valid_ids) in enumerate(kfold.split(total_train_set)):
        train_subsampler = SubsetRandomSampler(train_ids)
        valid_subsampler = SubsetRandomSampler(valid_ids)

        train_dataloader = DataLoader(total_train_set, batch_size=batch_size, sampler=train_subsampler)
        test_dataloader = DataLoader(total_train_set, batch_size=1, sampler=valid_subsampler)

        model = Net_NN(dropout_rate)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        # training
        for epoch in range(nb_epochs):
            train(train_dataloader, model, loss_fn, optimizer)
            print("Epoch", epoch + 1, "finished.")

        # save trained model to a file
        path = f'results/{dataset}/kfold/{model_file_name_dir}'
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f'results/{dataset}/kfold/{model_file_name_dir}/fold_{fold_nb}', "wb+") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # testing
        fold_accuracy = test(test_dataloader, model)
        accuracies.append(fold_accuracy)
        print(fold_nb + 1, "-- Fold accuracy: ", fold_accuracy)

    return accuracies, sum(accuracies) / 10

################################################# DATASET ###############################################
# dataset = "mnist"
dataset = "fashion_mnist"
#########################################################################################################

############################################### PARAMETERS ##############################################
seed = 0
nb_epochs = 1
batch_size = 8
learning_rate = 0.001
dropout_rate = 0
#########################################################################################################

# setting seeds for reproducibility
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)

# generate and shuffle dataset
if dataset == "mnist":
    generate_dataset_mnist(seed, 0)
    processed_data_path = "../data/MNIST/processed/"
elif dataset == "fashion_mnist":
    generate_dataset_fashion_mnist(seed, 0)
    processed_data_path = "../data/FashionMNIST/processed/"

# import train set
train_set = parse_data(dataset, processed_data_path)

# generate name of folder that holds all the trained models
model_file_name_dir = "NN_kfold_{}_{}_{}_{}_{}".format(seed, nb_epochs, batch_size, learning_rate, 
    dropout_rate)

# train and test
accuracies, avg_accuracy = train_and_test(dataset, model_file_name_dir, train_set, nb_epochs, 
    batch_size, learning_rate, dropout_rate)

# save results to a summary file
information = {
    "algorithm": "NN",
    "seed": seed,
    "nb_epochs": nb_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "dropout_rate": dropout_rate,
    "accuracies": accuracies,
    "avg_accuracy": avg_accuracy,
    "model_files_dir": model_file_name_dir
}
with open(f'results/{dataset}/kfold/summary_kfold.json', "a") as outfile:
    json.dump(information, outfile)
    outfile.write('\n')

# print results
print("############################################")
print("Seed: {} \nAvg_accuracy: {}".format(seed, avg_accuracy))
print("############################################")