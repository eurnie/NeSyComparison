import random
import numpy
import time
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

sys.path.append("..")
from data.generate_dataset import generate_dataset
from data.network_torch import Net_SL, Net_SL_Dropout

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
        new_tensor = numpy.array([torch.cat((datasets[dataset_used][index_digit_1][0][0], datasets[dataset_used][index_digit_2][0][0]), 0).numpy()])

        if (dataset_name == "train"):
            dataset.append((torch.tensor(new_tensor), [datasets[dataset_name][index_digit_1][1], datasets[dataset_name][index_digit_2][1]+10]))
        elif ((dataset_name == "test") or (dataset_name == "val")):
            sum = int(entries[i].split(" ")[2])
            dataset.append((torch.tensor(new_tensor), sum))

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
            correct += (result == y).type(torch.float).sum().item()
            total += len(x)
    return correct / total 

def train_and_test(model_file_name, train_set, val_set, test_set, nb_epochs, batch_size, learning_rate, 
    use_dropout):
    if use_dropout:
        model = Net_SL_Dropout()
    else:
        model = Net_SL()
    sl = SemanticLoss('constraint.sdd', 'constraint.vtree')
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataloader = DataLoader(train_set, batch_size=batch_size)
    val_dataloader = DataLoader(val_set, batch_size=1)
    test_dataloader = DataLoader(test_set, batch_size=1)

    # display image and label
    # train_features, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    # img = train_features[0].squeeze()
    # label = train_labels[0]
    # plt.imshow(img, cmap="gray")
    # plt.show()
    # print(f"Label: {label}")

    # training (with early stopping)
    total_training_time = 0
    best_accuracy = 0
    counter = 0
    for epoch in range(nb_epochs):
        start_time = time.time()
        train(train_dataloader, model, sl, loss_fn, optimizer)
        total_training_time += time.time() - start_time
        val_accuracy = test(val_dataloader, model)
        print("Val accuracy after epoch", epoch, ":", val_accuracy)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            with open("best_model.pickle", "wb") as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            counter = 0
        else:
            if counter >= 0:
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
nb_epochs = 4
batch_size = 4
learning_rate = 0.001
use_dropout = True
size_val = 0.1
#########################################################################################################

for seed in range(0, 10):
    # setting seeds for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    # shuffle dataset
    generate_dataset(seed)

    # import train, val and test set
    train_set = parse_data("../data/MNIST/processed/train.txt", "train", size_val)
    val_set = parse_data("../data/MNIST/processed/train.txt", "val", size_val)
    test_set = parse_data("../data/MNIST/processed/test.txt", "test", size_val)

    # generate name of file that holds the trained model
    model_file_name = "SL_final_{}_{}_{}_{}_{}_{}".format(seed, nb_epochs, batch_size, learning_rate, 
        use_dropout, size_val)

    # train and test
    accuracy, training_time, testing_time = train_and_test(model_file_name, train_set, val_set, test_set, 
        nb_epochs, batch_size, learning_rate, use_dropout)
    
    # save results to a summary file
    information = {
        "algorithm": "SL",
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