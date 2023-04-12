import random
import numpy
import sys
import os
import json
import torch
import torchvision
import pickle
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from semantic_loss_pytorch import SemanticLoss
from pathlib import Path
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler

sys.path.append("../..")
from data.generate_dataset import generate_dataset_mnist, generate_dataset_fashion_mnist
from data.network_torch import Net_SL, Net_SL_Dropout

def parse_data(dataset, filename):
    DATA_ROOT = Path(__file__).parent.parent.parent.joinpath('data')

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
        new_tensor = numpy.array([torch.cat((datasets[dataset_used][index_digit_1][0][0], datasets[dataset_used][index_digit_2][0][0]), 0).numpy()])
        dataset.append((torch.tensor(new_tensor), [datasets[dataset_used][index_digit_1][1], datasets[dataset_used][index_digit_2][1]+10]))

    return dataset

def train(dataloader, model, sl, loss_fn, optimizer):
    model.train()
    for (x, y) in dataloader:
        # compute prediction error
        multilabels = []
        for i in range(0, len(x)):
            new_label = numpy.zeros(20)
            new_label[y[0][i]] = 1
            new_label[y[1][i]] = 1
            multilabels.append(new_label)

        pred = model(x)
        loss = loss_fn(pred, torch.tensor(numpy.array(multilabels)).type(torch.float)) + sl(pred)

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
            predicted = torch.topk(pred, 2, largest=True).indices.numpy()[0]
            if predicted[0] < predicted[1]:
                number_1 = predicted[0]
                number_2 = predicted[1] - 10
            else:
                number_1 = predicted[0] - 10
                number_2 = predicted[1]
            result = number_1 + number_2
            real_number_1 = y[0]
            real_number_2 = y[1] - 10
            label = real_number_1 + real_number_2
            correct += (result == label).type(torch.float).sum().item()
            total += len(x)
    return correct / total 

def train_and_test(dataset, model_file_name_dir, total_train_set, nb_epochs, batch_size, learning_rate, 
    use_dropout):
    accuracies = []
    kfold = KFold(n_splits=10, shuffle=True)

    for fold_nb, (train_ids, valid_ids) in enumerate(kfold.split(total_train_set)):
        train_subsampler = SubsetRandomSampler(train_ids)
        valid_subsampler = SubsetRandomSampler(valid_ids)

        train_dataloader = DataLoader(total_train_set, batch_size=batch_size, sampler=train_subsampler)
        test_dataloader = DataLoader(total_train_set, batch_size=1, sampler=valid_subsampler)

        if use_dropout:
            model = Net_SL_Dropout()
        else:
            model = Net_SL()
        sl = SemanticLoss('constraint.sdd', 'constraint.vtree')
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
        for epoch in range(nb_epochs):
            train(train_dataloader, model, sl, loss_fn, optimizer)
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
batch_size = 2
learning_rate = 0.001
use_dropout = False
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
model_file_name_dir = "SL_kfold_{}_{}_{}_{}_{}".format(seed, nb_epochs, batch_size, learning_rate, 
    use_dropout)

# train and test
accuracies, avg_accuracy = train_and_test(dataset, model_file_name_dir, train_set, nb_epochs, 
    batch_size, learning_rate, use_dropout)

# save results to a summary file
information = {
    "algorithm": "SL",
    "seed": seed,
    "nb_epochs": nb_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "use_dropout": use_dropout,
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