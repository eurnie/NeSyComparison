import random
import numpy
import time
import sys
import os
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

def train_and_test(dataset, label_noise, model_file_name, train_set, val_set, 
                   test_set, nb_epochs, batch_size, learning_rate, dropout_rate):
    # create model and loss functions
    model = Net(dropout_rate)
    sl = []
    for sum in range(19):
        sl.append(SemanticLoss(f'constraints/sum_{sum}/constraint.sdd', f'constraints/sum_{sum}/constraint.vtree'))
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
        train(train_dataloader, model, sl, optimizer)
        total_training_time += time.time() - start_time
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
    with open(f'results/{dataset}/final/label_noise_{label_noise}/{model_file_name}', "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    # testing
    start_time = time.time()
    accuracy = test(test_dataloader, model)
    testing_time = time.time() - start_time

    return nb_epochs_done, accuracy, total_training_time, testing_time

################################################# DATASET ###############################################
dataset = "mnist"
# dataset = "fashion_mnist"
label_noise = 0.5
#########################################################################################################

############################################### PARAMETERS ##############################################
nb_epochs = 100
batch_size = 32
learning_rate = 0.001
dropout_rate = 0
size_val = 0.1
#########################################################################################################

for seed in range(0, 10):
    # setting seeds for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    # generate and shuffle dataset
    if dataset == "mnist":
        generate_dataset_mnist(seed, label_noise)
        processed_data_path = "../data/MNIST/processed/"
    elif dataset == "fashion_mnist":
        generate_dataset_fashion_mnist(seed, label_noise)
        processed_data_path = "../data/FashionMNIST/processed/"

    # import train, val and test set
    train_set = parse_data(dataset, processed_data_path, "train", size_val)
    val_set = parse_data(dataset, processed_data_path, "val", size_val)
    test_set = parse_data(dataset, processed_data_path, "test", size_val)

    # generate name of file that holds the trained model
    model_file_name = "SL_final_{}_{}_{}_{}_{}_{}".format(label_noise, seed, 
        nb_epochs, batch_size, learning_rate, dropout_rate, size_val)

    # train and test
    nb_epochs_done, accuracy, training_time, testing_time = train_and_test(dataset, label_noise, 
                                model_file_name, train_set, val_set, test_set, nb_epochs, batch_size, 
                                learning_rate, dropout_rate)
    
    # save results to a summary file
    information = {
        "algorithm": "SL",
        "seed": seed,
        "nb_epochs": nb_epochs_done,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dropout_rate": dropout_rate,
        "size_val": size_val,
        "accuracy": accuracy,
        "training_time": training_time,
        "testing_time": testing_time,
        "model_file": model_file_name
    }
    with open(f'results/{dataset}/final/label_noise_{label_noise}/summary_final.json', "a") as outfile:
        json.dump(information, outfile)
        outfile.write('\n')

    # print results
    print("############################################")
    print("Seed: {} \nAccuracy: {} \nTraining time: {} \nTesting time: {}".format(seed, accuracy, 
        training_time, testing_time))
    print("############################################")