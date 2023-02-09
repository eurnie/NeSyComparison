import random
import numpy
import time
import sys
import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from semantic_loss_pytorch import SemanticLoss
from pathlib import Path

sys.path.append("..")
from data.generate_dataset import generate_dataset
from data.network_torch import Net_SL

def parse_data(filename, dataset_name):
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

    with open(filename) as f:
        entries = f.readlines()

    dataset = []

    for entry in entries:
        index_digit_1 = int(entry.split(" ")[0])
        index_digit_2 = int(entry.split(" ")[1])
        new_tensor = [torch.cat((datasets[dataset_name][index_digit_1][0][0], datasets[dataset_name][index_digit_2][0][0]), 0).numpy()]

        if (dataset_name == "train"):
            dataset.append((torch.tensor(new_tensor), [datasets[dataset_name][index_digit_1][1], datasets[dataset_name][index_digit_2][1]+10]))
        elif (dataset_name == "test"):
            sum = int(entry.split(" ")[2])
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
        loss = loss_fn(pred, torch.tensor(multilabels).type(torch.float)) + sl(pred)

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
            predicted = torch.topk(pred, 2, largest=True).indices.numpy()[0]
            number_1 = predicted[0]
            number_2 = predicted[1] - 10
            result = number_1 + number_2
            correct += (result == y).type(torch.float).sum().item()
    correct /= size
    return 100*correct

def train_and_test(train_set, test_set, nb_epochs, batch_size, learning_rate):
    model = Net_SL()
    sl = SemanticLoss('constraint.sdd', 'constraint.vtree')
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataloader = DataLoader(train_set, batch_size=batch_size)
    test_dataloader = DataLoader(test_set, batch_size=1)

    # Display image and label.
    # train_features, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    # img = train_features[0].squeeze()
    # label = train_labels[0]
    # plt.imshow(img, cmap="gray")
    # plt.show()
    # print(f"Label: {label}")

    # training
    start_time = time.time()
    for _ in range(0, nb_epochs):          
        train(train_dataloader, model, sl, loss_fn, optimizer)
    training_time = time.time() - start_time
            
    # testing
    accuracy = test(test_dataloader, model)

    return accuracy, training_time

############################################### PARAMETERS ##############################################
nb_epochs = 3
batch_size = 8
learning_rate = 0.001
#########################################################################################################

for seed in range(0, 10):
    # setting seeds for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    # import train and test set (shuffled)
    generate_dataset(seed)
    train_set = parse_data("../data/MNIST/processed/train.txt", "train")
    test_set = parse_data("../data/MNIST/processed/test.txt", "test")

    # train and test the method on the MNIST addition dataset
    accuracy, training_time = train_and_test(train_set, test_set, nb_epochs, batch_size, learning_rate)

    # print results
    print("############################################")
    print("Seed: {} \nAccuracy: {} \nTraining time: {}".format(seed, accuracy, training_time))
    print("############################################")