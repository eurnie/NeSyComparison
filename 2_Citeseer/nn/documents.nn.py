import sys
import random
import numpy
import time
import torch
import torch.nn.functional as F
import torch_geometric
from pathlib import Path

sys.path.append("..")
from data.network_torch import Net_NN

def train_and_test(dataset, nb_epochs, learning_rate):
    model = Net_NN()
    data = dataset[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training
    start_time = time.time()
    for _ in range(0, nb_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    training_time = time.time() - start_time

    # testing
    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    accuracy = int(correct) / int(data.test_mask.sum())

    return accuracy, training_time

############################################### PARAMETERS ##############################################
nb_epochs = 10
learning_rate = 0.001
#########################################################################################################

for seed in range(0, 10):
    # setting seeds for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    # import train and test set
    DATA_ROOT = Path(__file__).parent.parent.joinpath('data')
    dataset = torch_geometric.datasets.Planetoid(root=str(DATA_ROOT), name="CiteSeer", split="full")

    # train and test the method on the MNIST addition dataset
    accuracy, training_time = train_and_test(dataset, nb_epochs, learning_rate)

    # print results
    print("############################################")
    print("Seed: {} \nAccuracy: {} \nTraining time: {}".format(seed, accuracy, training_time))
    print("############################################")