import random
import numpy
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch
from pathlib import Path
from network import Net
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path

############################################################################################
SEED_PYTHON = 123
SEED_NUMPY = 456
SEED_TORCH = 789
batch_size = 10
nb_epochs = 3
learning_rate = 1e-3
############################################################################################
log_iter = 1000
############################################################################################

# setting seeds for reproducibility
random.seed(SEED_PYTHON)
numpy.random.seed(SEED_NUMPY)
torch.manual_seed(SEED_TORCH)

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

def parse_data(filename, dataset_name):
    with open(filename) as f:
        entries = f.readlines()

    dataset = []

    for entry in entries:
        index_digit_1 = int(entry.split(" ")[0])
        index_digit_2 = int(entry.split(" ")[1])
        sum = int(entry.split(" ")[2])
        dataset.append((torch.cat((datasets[dataset_name][index_digit_1][0], datasets[dataset_name][index_digit_2][0]), 0), sum))
        
    return dataset

model = Net()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

training_set = parse_data("../data/MNIST/processed/train.txt", "train")
testing_set = parse_data("../data/MNIST/processed/test.txt", "test")

nb_training_examples = len(training_set)

train_dataloaders = []

for i in range(0, nb_training_examples + log_iter, log_iter):
    train_dataloaders.append(DataLoader(training_set[i:i+log_iter], batch_size=batch_size))

test_dataloader = DataLoader(testing_set, batch_size=batch_size)

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

total_training_time = 0
highest_accuracy = 0
highest_accuracy_index = 0

for epoch in range(0, nb_epochs):
    for i in range(0, len(train_dataloaders)):

        # training
        start_time = time.time()
        train(train_dataloaders[i], model, loss_fn, optimizer)
        total_training_time += time.time() - start_time

        # testing
        accuracy = test(test_dataloader, model)

        if (accuracy > highest_accuracy):
            highest_accuracy = accuracy
            highest_accuracy_index = (epoch * nb_training_examples) + log_iter + (i * log_iter)

        log_file = "results/results_nn_{}_{}_{}_{}_{}_{}.txt".format(SEED_PYTHON, SEED_NUMPY, SEED_TORCH, batch_size, nb_epochs, learning_rate)

        with open(log_file, "a") as f:
            f.write(str((epoch * nb_training_examples) + log_iter + (i * log_iter)))
            f.write(" ")
            f.write(str(total_training_time))
            f.write(" ")
            f.write(str(accuracy))
            f.write(" ")
            f.write("\n")

        print("############################################")
        print("Number of entries: ", (epoch * nb_training_examples) + log_iter + (i * log_iter))
        print("Total training time: ", total_training_time)
        print("Accuracy: ", accuracy)
        print("############################################")

print("The highest accuracy was {} and was reached (the first time) after seeing {} samples.".format(highest_accuracy, highest_accuracy_index))