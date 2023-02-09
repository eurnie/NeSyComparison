import os
import sys
import time
import random
import numpy
import torch
import torchvision
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from neurasp.neurasp import NeurASP

sys.path.append("..")
from data.generate_dataset import generate_dataset
from data.network_torch import Net

class MNIST_Addition(Dataset):
    def __init__(self, dataset, examples):
        self.data = list()
        self.dataset = dataset
        with open(examples) as f:
            for line in f:
                line = line.strip().split(' ')
                self.data.append(tuple([int(i) for i in line]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        i1, i2, l = self.data[index]
        return self.dataset[i1][0], self.dataset[i2][0], l

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

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
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
    )
}

# NeurASP program
dprogram = '''
img(i1). img(i2).
addition(A,B,N) :- digit(0,A,N1), digit(0,B,N2), N=N1+N2.
nn(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).
'''

for seed in range(0, 10):
    # setting seeds for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    # import train and test set (shuffled)
    generate_dataset(seed)

    trainDataset = MNIST_Addition(datasets["train"], dir_path + "/../data/MNIST/processed/train.txt")
    dataList_train = []
    obsList_train = []
    for i1, i2, l in trainDataset:
        dataList_train.append({'i1': i1[0].unsqueeze(0), 'i2': i2[0].unsqueeze(0)})
        obsList_train.append(':- not addition(i1, i2, {}).'.format(l))

    testDataset = MNIST_Addition(datasets["test"], dir_path + "/../data/MNIST/processed/test.txt")
    dataList_test = []
    obsList_test = []
    for i1, i2, l in testDataset:
        dataList_test.append({'i1': i1[0].unsqueeze(0), 'i2': i2[0].unsqueeze(0)})
        obsList_test.append(':- not addition(i1, i2, {}).'.format(l))

    # define nnMapping and optimizers, initialze NeurASP object
    m = Net()
    nnMapping = {'digit': m}
    optimizers = {'digit': torch.optim.Adam(m.parameters(), lr=learning_rate)}
    NeurASPobj = NeurASP(dprogram, nnMapping, optimizers)

    # train and test the method on the MNIST addition dataset
    accuracy, training_time = train_and_test(dataList_train, obsList_train, dataList_test, obsList_test,
        nb_epochs)

    # print results
    print("############################################")
    print("Seed: {} \nAccuracy: {} \nTraining time: {}".format(seed, accuracy, training_time))
    print("############################################")