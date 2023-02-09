import os
import sys
import time
import random
import numpy
import torch
import torch_geometric
from pathlib import Path
from torch.utils.data import Dataset
from neurasp_program import dprogram
from neurasp.neurasp import NeurASP

sys.path.append("..")
from data.network_torch import Net

class Citeseer(Dataset):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.data = []

        for i in range(len(dataset.y)):
            if self.dataset_name == "train":
                if dataset.train_mask[i]:
                    self.data.append(str(i))
            elif self.dataset_name == "val":
                if dataset.val_mask[i]:
                    self.data.append(str(i))
            elif self.dataset_name == "test":
                if dataset.test_mask[i]:
                    self.data.append(str(i))
        
        self.labels = dataset.y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def train_and_test(dataList_train, obsList_train, dataList_test, obsList_test, nb_epochs):
    start_time = time.time()
    NeurASPobj.learn(dataList=dataList_train, obsList=obsList_train, epoch=nb_epochs, smPickle=None, bar=True)
    training_time = time.time() - start_time

    accuracy = NeurASPobj.testInferenceResults(dataList_test, obsList_test) / 100

    return accuracy, training_time

############################################### PARAMETERS ##############################################
nb_epochs = 1
learning_rate = 0.001
#########################################################################################################

DATA_ROOT = Path(__file__).parent.parent.joinpath('data')
dataset = torch_geometric.datasets.Planetoid(root=str(DATA_ROOT), name="CiteSeer", split="full")[0]

for seed in range(0, 10):
    # setting seeds for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    trainDataset = Citeseer("train")
    dataList_train = []
    obsList_train = []
    for index in trainDataset:
        dataList_train.append({'i1': dataset.x[int(index)]})
        dataList_train.append({'i1': dataset.x[int(index)]})
        obsList_train.append(':- not label({}, i1, {}).'.format(index, dataset.y[int(index)]))
        obsList_train.append(':- not index(i1, {}).'.format(index))

    testDataset = Citeseer("test")
    dataList_test = []
    obsList_test = []
    for index in testDataset:
        dataList_test.append({'i1': dataset.x[int(index)]})
        dataList_test.append({'i1': dataset.x[int(index)]})
        obsList_test.append(':- not label(i1, {}).'.format(dataset.y[int(index)]))
        obsList_test.append(':- not index(i1, {}).'.format(index))

    # define nnMapping and optimizers, initialze NeurASP object
    m = Net()
    nnMapping = {'label': m}
    optimizers = {'label': torch.optim.Adam(m.parameters(), lr=learning_rate)}
    NeurASPobj = NeurASP(dprogram, nnMapping, optimizers)

    # train and test the method on the MNIST addition dataset
    accuracy, training_time = train_and_test(dataList_train, obsList_train, dataList_test, obsList_test,
        nb_epochs)

    # print results
    print("############################################")
    print("Seed: {} \nAccuracy: {} \nTraining time: {}".format(seed, accuracy, training_time))
    print("############################################")