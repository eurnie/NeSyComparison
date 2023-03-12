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
from data.generate_dataset import generate_dataset
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

def train_and_test(model_file_name_dir, dataList_train_total, obsList_train_total, nb_epochs, batch_size):
    accuracies = []

    for fold_nb in range(1, 11):
        # define nnMapping and optimizers, initialze NeurASP object
        if use_dropout:
            m = Net_Dropout()
        else:
            m = Net()
        nnMapping = {'digit': m}
        optimizers = {'digit': torch.optim.Adam(m.parameters(), lr=learning_rate)}
        NeurASPobj = NeurASP(dprogram, nnMapping, optimizers)

        # training
        for epoch in range(nb_epochs):
            for i in range(0, 10):
                if (i != (fold_nb - 1)):
                    dataList_train = dataList_train_total[i]
                    obsList_train = obsList_train_total[i]
                    NeurASPobj.learn(dataList=dataList_train, obsList=obsList_train, epoch=1, smPickle=None, 
                        bar=True, batchSize=batch_size)
                else:
                    dataList_test = dataList_train_total[i]
                    obsList_test = obsList_train_total[i]
            print("Epoch", epoch + 1, "finished.")
            
        # save trained model to a file
        path = "results/param/{}".format(model_file_name_dir)
        if not os.path.exists(path):
            os.makedirs(path)
        with open("results/param/{}/fold_{}".format(model_file_name_dir, fold_nb), "wb+") as handle:
            pickle.dump(NeurASPobj, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # testing
        fold_accuracy = NeurASPobj.testInferenceResults(dataList_test, obsList_test) / 100
        accuracies.append(fold_accuracy)
        print(fold_nb, "-- Fold accuracy: ", fold_accuracy)

    return accuracies, sum(accuracies) / 10

############################################### PARAMETERS ##############################################
seed = 0
nb_epochs = 2
batch_size = 16
learning_rate = 0.001
use_dropout = True
#########################################################################################################

# (2, 16, 0.001, True)
# (2, 8, 0.001, False)
# (3, 32, 0.001, True)
# (1, 32, 0.001, False)

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

# setting seeds for reproducibility
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)

# shuffle dataset
generate_dataset(seed)

# import train set
dataList_train_total = []
obsList_train_total = []
for i in range(0, 30000, 3000):
    trainDataset = MNIST_Addition(datasets["train"], dir_path + "/../data/MNIST/processed/train.txt",
        start_index=i, end_index=i+3000)
    dataList_train = []
    obsList_train = []
    for i1, i2, l in trainDataset:
        dataList_train.append({'i1': i1[0].unsqueeze(0), 'i2': i2[0].unsqueeze(0)})
        obsList_train.append(':- not addition(i1, i2, {}).'.format(l))
    dataList_train_total.append(dataList_train)
    obsList_train_total.append(obsList_train)

# generate name of folder that holds all the trained models
model_file_name_dir = "NeurASP_param_{}_{}_{}_{}_{}".format(seed, nb_epochs, batch_size, learning_rate, 
    use_dropout)

# train and test the method on the MNIST addition dataset
accuracies, avg_accuracy = train_and_test(model_file_name_dir, dataList_train_total, 
    obsList_train_total, nb_epochs, batch_size)

# save results to a summary file
information = {
    "algorithm": "NeurASP",
    "seed": seed,
    "nb_epochs": nb_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "use_dropout": use_dropout,
    "accuracies": accuracies,
    "avg_accuracy": avg_accuracy,
    "model_files_dir": model_file_name_dir
}
with open("results/summary_param.json", "a") as outfile:
    json.dump(information, outfile)
    outfile.write('\n')

# print results
print("############################################")
print("Seed: {} \nAccuracy: {}".format(seed, avg_accuracy))
print("############################################")