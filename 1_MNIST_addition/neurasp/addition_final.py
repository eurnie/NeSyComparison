import os
import json
import sys
import time
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

def train_and_test(model_file_name, dataList_train, obsList_train, dataList_val, obsList_val, 
    dataList_test, obsList_test, nb_epochs, batch_size):
    
    # training (with early stopping)
    total_training_time = 0
    best_accuracy = 0
    counter = 0

    for epoch in range(nb_epochs):
        start_time = time.time()
        NeurASPobj.learn(dataList=dataList_train, obsList=obsList_train, epoch=1, smPickle=None, 
            bar=True, batchSize=batch_size)
        total_training_time += time.time() - start_time
        val_accuracy = NeurASPobj.testInferenceResults(dataList_val, obsList_val) / 100
        print("Val accuracy after epoch", epoch, ":", val_accuracy)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            with open("best_model.pickle", "wb") as handle:
                pickle.dump(NeurASPobj, handle, protocol=pickle.HIGHEST_PROTOCOL)
            counter = 0
        else:
            if counter >= 1:
                break
            counter += 1
    with open("best_model.pickle", "rb") as handle:
        BestNeurASPobj = pickle.load(handle)

    os.remove("best_model.pickle")

    # save trained model to a file
    with open("results/final/{}".format(model_file_name), "wb") as handle:
        pickle.dump(NeurASPobj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # testing
    start_time = time.time()
    accuracy = BestNeurASPobj.testInferenceResults(dataList_test, obsList_test) / 100
    testing_time = time.time() - start_time

    return accuracy, total_training_time, testing_time

############################################### PARAMETERS ##############################################
nb_epochs = 3
batch_size = 8
learning_rate = 0.001
use_dropout = True
size_val = 0.1
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

    # shuffle dataset
    generate_dataset(seed)

    # import train, val and test set
    split_index = round(size_val * 30000)

    trainDataset = MNIST_Addition(datasets["train"], dir_path + "/../data/MNIST/processed/train.txt",
        start_index=split_index, end_index=30000)
    dataList_train = []
    obsList_train = []
    for i1, i2, l in trainDataset:
        dataList_train.append({'i1': i1[0].unsqueeze(0), 'i2': i2[0].unsqueeze(0)})
        obsList_train.append(':- not addition(i1, i2, {}).'.format(l))

    valDataset = MNIST_Addition(datasets["train"], dir_path + "/../data/MNIST/processed/train.txt",
        start_index=0, end_index=split_index)
    dataList_val = []
    obsList_val = []
    for i1, i2, l in valDataset:
        dataList_val.append({'i1': i1[0].unsqueeze(0), 'i2': i2[0].unsqueeze(0)})
        obsList_val.append(':- not addition(i1, i2, {}).'.format(l))

    testDataset = MNIST_Addition(datasets["test"], dir_path + "/../data/MNIST/processed/test.txt",
        start_index=0, end_index=5000)
    dataList_test = []
    obsList_test = []
    for i1, i2, l in testDataset:
        dataList_test.append({'i1': i1[0].unsqueeze(0), 'i2': i2[0].unsqueeze(0)})
        obsList_test.append(':- not addition(i1, i2, {}).'.format(l))

    # define nnMapping and optimizers, initialze NeurASP object
    if use_dropout:
        m = Net_Dropout()
    else:
        m = Net()
    nnMapping = {'digit': m}
    optimizers = {'digit': torch.optim.Adam(m.parameters(), lr=learning_rate)}
    NeurASPobj = NeurASP(dprogram, nnMapping, optimizers)

    # generate name of file that holds the trained model
    model_file_name = "NeurASP_final_{}_{}_{}_{}_{}_{}".format(seed, nb_epochs, batch_size, 
        learning_rate, use_dropout, size_val)

    # train and test the method on the MNIST addition dataset
    accuracy, training_time, testing_time = train_and_test(model_file_name, dataList_train, obsList_train,
        dataList_val, obsList_val, dataList_test, obsList_test, nb_epochs, batch_size)

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