import os
import json
import sys
import random
import numpy
import torch
import torchvision
import pickle
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from neurasp.neurasp import NeurASP

sys.path.append("..")
from data.generate_dataset import generate_dataset_mnist, generate_dataset_fashion_mnist
from data.network_torch import Net, Net_Dropout

class MNIST_Addition(Dataset):
    def __init__(self, dataset, examples, start_index, end_index):
        self.data = list()
        self.dataset = dataset
        with open(examples) as f:
            entries = f.readlines()
        for i in range(start_index, end_index):
            line = entries[i]
            line = line.strip().split(' ')
            self.data.append(tuple([int(i) for i in line]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        i1, i2, l = self.data[index]
        return self.dataset[i1][0], self.dataset[i2][0], l

def train_and_test(dataset, model_file_name, dataList_train, obsList_train, 
    dataList_val, obsList_val, nb_epochs, batch_size):
    
    NeurASPobj.learn(dataList=dataList_train, obsList=obsList_train, epoch=nb_epochs, smPickle=None, 
        bar=True, batchSize=batch_size)

    # save trained model to a file
    with open(f'results/{dataset}/{model_file_name}', "wb") as handle:
        pickle.dump(NeurASPobj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # testing
    accuracy = NeurASPobj.testInferenceResults(dataList_val, obsList_val) / 100
    return accuracy

################################################# DATASET ###############################################
dataset = "mnist"
# dataset = "fashion_mnist"
#########################################################################################################

############################################### PARAMETERS ##############################################
seed = 0
nb_epochs = 5
batch_size = 32
learning_rate = 0.001
use_dropout = False
size_val = 0.1
#########################################################################################################

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
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
    )
}

datasets_fashion_mnist = {
    "train": torchvision.datasets.FashionMNIST(
        root=str(DATA_ROOT), train=True, download=True, transform=transform
    ),
    "test": torchvision.datasets.FashionMNIST(
        root=str(DATA_ROOT), train=False, download=True, transform=transform
    )
}

# NeurASP program
dprogram = '''
img(i1). img(i2).
addition(A,B,N) :- digit(0,A,N1), digit(0,B,N2), N=N1+N2.
nn(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).
'''

# setting seeds for reproducibility
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)

# generate and shuffle dataset
split_index = round(size_val * 30000)

if dataset == "mnist":
    generate_dataset_mnist(seed, 0)
    trainDataset = MNIST_Addition(datasets_mnist["train"], dir_path + "/../data/MNIST/processed/train.txt",
        start_index=split_index, end_index=30000)
    valDataset = MNIST_Addition(datasets_mnist["train"], dir_path + "/../data/MNIST/processed/train.txt",
        start_index=0, end_index=split_index)
elif dataset == "fashion_mnist":
    generate_dataset_fashion_mnist(seed, 0)
    trainDataset = MNIST_Addition(datasets_fashion_mnist["train"], dir_path + "/../data/FashionMNIST/processed/train.txt",
        start_index=split_index, end_index=30000)
    valDataset = MNIST_Addition(datasets_fashion_mnist["train"], dir_path + "/../data/FashionMNIST/processed/train.txt",
        start_index=0, end_index=split_index)

dataList_train = []
obsList_train = []
for i1, i2, l in trainDataset:
    dataList_train.append({'i1': i1[0].unsqueeze(0), 'i2': i2[0].unsqueeze(0)})
    obsList_train.append(':- not addition(i1, i2, {}).'.format(l))

dataList_val = []
obsList_val = []
for i1, i2, l in valDataset:
    dataList_val.append({'i1': i1[0].unsqueeze(0), 'i2': i2[0].unsqueeze(0)})
    obsList_val.append(':- not addition(i1, i2, {}).'.format(l))

# define nnMapping and optimizers, initialze NeurASP object
if use_dropout:
    m = Net_Dropout()
else:
    m = Net()
nnMapping = {'digit': m}
optimizers = {'digit': torch.optim.Adam(m.parameters(), lr=learning_rate)}
NeurASPobj = NeurASP(dprogram, nnMapping, optimizers)

# generate name of file that holds the trained model
model_file_name = "param/NeurASP_param_{}_{}_{}_{}_{}_{}".format(seed, 
    nb_epochs, batch_size, learning_rate, use_dropout, size_val)

# train and test the method on the MNIST addition dataset
accuracy = train_and_test(dataset, model_file_name, dataList_train, obsList_train, 
                          dataList_val, obsList_val, nb_epochs, batch_size)

# save results to a summary file
information = {
    "algorithm": "NeurASP",
    "seed": seed,
    "nb_epochs": nb_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "use_dropout": use_dropout,
    "size_val": size_val,
    "accuracy": accuracy,
    "model_file": model_file_name
}
with open(f'results/{dataset}/param/summary_param.json', "a") as outfile:
    json.dump(information, outfile)
    outfile.write('\n')

# print results
print("############################################")
print("Accuracy: {}".format(accuracy))
print("############################################")
