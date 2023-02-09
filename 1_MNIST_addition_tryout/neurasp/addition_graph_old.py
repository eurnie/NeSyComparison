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
from network import Net
from neurasp.neurasp import NeurASP

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
sys.path.append('../../')

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

############################################################################################
SEED_PYTHON = 123
SEED_NUMPY = 456
SEED_TORCH = 789
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
    )
}

trainDataset = MNIST_Addition(datasets["train"], dir_path + "/../data/MNIST/processed/train.txt")
completeDataList = []
completeObsList = []
for i in range(0, len(trainDataset), log_iter):
    dataList = []
    obsList = []
    for j in range(0, log_iter):
        i1 = trainDataset[i+j][0]
        i2 = trainDataset[i+j][1]
        l = trainDataset[i+j][2]

        dataList.append({'i1': i1[0].unsqueeze(0), 'i2': i2[0].unsqueeze(0)})
        obsList.append(':- not addition(i1, i2, {}).'.format(l))
    completeDataList.append(dataList)
    completeObsList.append(obsList)

testDataset = MNIST_Addition(datasets["test"], dir_path + "/../data/MNIST/processed/test.txt")
dataList_test = []
obsList_test = []
for i1, i2, l in testDataset:
    dataList_test.append({'i1': i1[0].unsqueeze(0), 'i2': i2[0].unsqueeze(0)})
    obsList_test.append(':- not addition(i1, i2, {}).'.format(l))

# NeurASP program
dprogram = '''
img(i1). img(i2).
addition(A,B,N) :- digit(0,A,N1), digit(0,B,N2), N=N1+N2.
nn(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).
'''

# define nnMapping and optimizers, initialze NeurASP object
m = Net()
nnMapping = {'digit': m}
optimizers = {'digit': torch.optim.Adam(m.parameters(), lr=learning_rate)}
NeurASPobj = NeurASP(dprogram, nnMapping, optimizers)

total_training_time = 0
highest_accuracy = 0
highest_accuracy_index = 0

for epoch in range(0, nb_epochs):
    for nb in range(0, len(completeDataList)):
        start_time = time.time()
        # print(completeDataList[nb][0])
        NeurASPobj.learn(dataList=completeDataList[nb], obsList=completeObsList[nb], epoch=1, smPickle=None, bar=True)
        total_training_time += time.time() - start_time

        accuracy = NeurASPobj.testInferenceResults(dataList_test, obsList_test) / 100

        if (accuracy > highest_accuracy):
            highest_accuracy = accuracy
            highest_accuracy_index = (epoch * 30000) + log_iter + (nb * log_iter)

        log_file = "results/results_neurasp_{}_{}_{}_{}_{}.txt".format(SEED_PYTHON, SEED_NUMPY, SEED_TORCH, nb_epochs, learning_rate)

        with open(log_file, "a") as f:
            f.write(str((epoch * 30000) + log_iter + (nb * log_iter)))
            f.write(" ")
            f.write(str(total_training_time))
            f.write(" ")
            f.write(str(accuracy))
            f.write(" ")
            f.write("\n")

        print("############################################")
        print("Number of entries: ", (epoch * 30000) + log_iter + (nb * log_iter))
        print("Total training time: ", total_training_time)
        print("Accuracy: ", accuracy)
        print("############################################")

print("The highest accuracy was {} and was reached (the first time) after seeing {} samples.".format(highest_accuracy, highest_accuracy_index))