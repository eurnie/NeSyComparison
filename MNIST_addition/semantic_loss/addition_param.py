import os
import random
import numpy
import sys
import json
import torch
import torchvision
import pickle
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from semantic_loss_pytorch import SemanticLoss
from pathlib import Path

sys.path.append("..")
from data.generate_dataset import generate_dataset_mnist, generate_dataset_fashion_mnist
from data.network_torch import Net

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

    if dataset == "mnist":
        datasets = datasets_mnist
    elif dataset == "fashion_mnist":
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
        first = datasets[dataset_used][index_digit_1][0]
        second = datasets[dataset_used][index_digit_2][0]
        dataset.append((first, second, sum))
    
    return dataset

def train(dataloader, model, sl, optimizer):
    model.train()
    for (img1, img2, y) in dataloader:
        # predict sum
        pred_digit_1 = model(img1)
        pred_digit_2 = model(img2)
        pred = torch.cat((pred_digit_1, pred_digit_2), 1)

        # calculate loss
        loss = 0
        for i, sum in enumerate(y):
            loss += sl[sum.item()](pred[i][None, :])

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(dataloader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img1, img2, y in dataloader:
            pred_digit_1 = model(img1)
            pred_digit_2 = model(img2)
            pred = pred_digit_1.argmax(1) + pred_digit_2.argmax(1)
            correct += (pred == y).type(torch.float).sum().item()
            total += len(img1)
    return correct / total

################################################# DATASET ###############################################
dataset = "mnist"
# dataset = "fashion_mnist"
#########################################################################################################

############################################### PARAMETERS ##############################################
seed = 0
nb_epochs = 100
size_val = 0.1
#########################################################################################################

for dropout_rate in [0, 0.2]:
    for optimizer_name in ["Adam", "SGD"]:
        for learning_rate in [0.001, 0.0001]:
            for batch_size in [2, 4, 8, 16, 32, 64]:
                # generate name of file that holds the trained model
                model_file_name = "SL_param_{}_{}_{}_{}_{}_{}_{}".format(seed, 
                    nb_epochs, size_val, dropout_rate, optimizer_name, learning_rate, batch_size)
                model_file_location = f'results/{dataset}/param/{model_file_name}'
                
                if not os.path.isfile(model_file_location):
                    # setting seeds for reproducibility
                    random.seed(seed)
                    numpy.random.seed(seed)
                    torch.manual_seed(seed)

                    # generate and shuffle dataset
                    if dataset == "mnist":
                        generate_dataset_mnist(seed, 0)
                    elif dataset == "fashion_mnist":
                        generate_dataset_fashion_mnist(seed, 0)
                    processed_data_path = f'../data/{dataset}/processed/'

                    # import train and val set
                    train_set = parse_data(dataset, processed_data_path, "train", size_val)
                    val_set = parse_data(dataset, processed_data_path, "val", size_val)

                    train_dataloader = DataLoader(train_set, batch_size=batch_size)
                    val_dataloader = DataLoader(val_set, batch_size=1)

                    # create model and loss functions
                    model = Net(dropout_rate)
                    sl = []
                    for sum in range(19):
                        sl.append(SemanticLoss(f'constraints/sum_{sum}/constraint.sdd', f'constraints/sum_{sum}/constraint.vtree'))
                    
                    if optimizer_name == "Adam":
                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                    elif optimizer_name == "SGD":
                        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

                    # training (with early stopping)
                    best_accuracy = 0
                    counter = 0
                    for epoch in range(nb_epochs):
                        train(train_dataloader, model, sl, optimizer)
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
                    with open(f'results/{dataset}/param/{model_file_name}', "wb") as handle:
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
                        "size_val": size_val,
                        "accuracy": best_accuracy,
                        "model_file": model_file_name
                    }
                    with open(f'results/{dataset}/summary_param.json', "a") as outfile:
                        json.dump(information, outfile)
                        outfile.write('\n')