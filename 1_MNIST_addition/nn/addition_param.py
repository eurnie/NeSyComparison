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
from data.generate_dataset import generate_dataset
from data.network_torch import Net_NN, Net_NN_Dropout, Net_NN_Extra

def parse_data(filename, dataset_name, size_val):
    DATA_ROOT = Path(__file__).parent.parent.joinpath('data')

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    datasets = {
        "train": torchvision.datasets.MNIST(
            root=str(DATA_ROOT), train=True, download=True, transform=transform
        ),
        "test": torchvision.datasets.MNIST(
            root=str(DATA_ROOT), train=False, download=True, transform=transform
        ),
    }

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

    with open(filename) as f:
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
    size = len(dataloader.dataset)
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct /= size
    return correct

def train_and_test(model_file_name_dir, total_train_set, nb_epochs, batch_size, learning_rate, use_dropout):
    accuracies = []
    kfold = KFold(n_splits=10, shuffle=True)

    for fold_nb, (train_ids, valid_ids) in enumerate(kfold.split(total_train_set)):
        train_subsampler = SubsetRandomSampler(train_ids)
        valid_subsampler = SubsetRandomSampler(valid_ids)

        train_dataloader = DataLoader(total_train_set, batch_size=batch_size, sampler=train_subsampler)
        test_dataloader = DataLoader(total_train_set, batch_size=1, sampler=valid_subsampler)

        if use_dropout:
            model = Net_NN_Dropout()
        else:
            model = Net_NN()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        # display image and label
        # train_features, train_labels = next(iter(train_dataloader))
        # print(f"Feature batch shape: {train_features.size()}")
        # print(f"Labels batch shape: {train_labels.size()}")
        # img = train_features[0].squeeze()
        # label = train_labels[0]
        # plt.imshow(img, cmap="gray")
        # plt.show()
        # print(f"Label: {label}")

        # training
        for _ in range(nb_epochs):
            train(train_dataloader, model, loss_fn, optimizer)

        # save trained model to a file
        path = "results/param/{}".format(model_file_name_dir)
        if not os.path.exists(path):
            os.makedirs(path)
        with open("results/param/{}/fold_{}".format(model_file_name_dir, fold_nb), "wb+") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # testing
        fold_accuracy = test(test_dataloader, model)
        accuracies.append(fold_accuracy)
        print(fold_nb, "-- Fold accuracy: ", fold_accuracy)

    return accuracies, sum(accuracies) / 10

############################################### PARAMETERS ##############################################
seed = 0
nb_epochs = 30
batch_size = 8
learning_rate = 0.001
use_dropout = True
#########################################################################################################

# (30, 8, 0.001, True)
# (50, 32, 0.001, True)
# (30, 16, 0.001, True)
# (10, 4, 0.001, True)
# (20, 16, 0.001, True)
# (40, 4, 0.001, True)
# (40, 2, 0.001, False)
# (30, 8, 0.001, False)
# (20, 32, 0.001, False)
# (40, 2, 0.001, True)

# setting seeds for reproducibility
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)

# shuffle dataset
generate_dataset(seed)

# import train set
train_set = parse_data("../data/MNIST/processed/train.txt", "train", 0)

# generate name of folder that holds all the trained models
model_file_name_dir = "NN_param_{}_{}_{}_{}_{}".format(seed, nb_epochs, batch_size, learning_rate, 
    use_dropout)

# train and test
accuracies, avg_accuracy = train_and_test(model_file_name_dir, train_set, nb_epochs, batch_size, 
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