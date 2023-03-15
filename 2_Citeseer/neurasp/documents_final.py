import os
import json
import sys
import time
import random
import numpy
import torch
import pickle
import torch_geometric
from pathlib import Path
from torch.utils.data import Dataset
from program import dprogram
from neurasp.neurasp import NeurASP

sys.path.append("..")
from data.network_torch import Net, Net_Dropout

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
nb_epochs = 1
batch_size = 16
learning_rate = 0.001
use_dropout = False
#########################################################################################################

DATA_ROOT = Path(__file__).parent.parent.joinpath('data')
data = torch_geometric.datasets.Planetoid(root=str(DATA_ROOT), name="CiteSeer", split="full")
citation_graph = data[0]

for seed in range(0, 10):
    # setting seeds for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    trainDataset = []
    valDataset = []
    testDataset = []

    # import train, val and test set
    for i in range(len(citation_graph.x)):
        if citation_graph.train_mask[i]:
            trainDataset.append((citation_graph.x[i], i, citation_graph.y[i]))
        elif citation_graph.val_mask[i]:
            valDataset.append((citation_graph.x[i], i, citation_graph.y[i]))
        elif citation_graph.test_mask[i]:
            testDataset.append((citation_graph.x[i], i, citation_graph.y[i]))

    dataList_train = []
    obsList_train = []
    for i1, i2, l in trainDataset:
        dataList_train.append({'i1': i1.unsqueeze(0), 'i2': i2})
        obsList_train.append(':- not document_type(0, i1, {}), not document_type_index(i2, {}).'.format(l, l))

    dataList_val = []
    obsList_val = []
    for i1, i2, l in valDataset:
        dataList_val.append({'i1': i1.unsqueeze(0), 'i2': i2})
        obsList_val.append(':- not document_type(0, i1, {}), not document_type_index(i2, {}).'.format(l, l))

    dataList_test = []
    obsList_test = []
    for i1, i2, l in testDataset:
        dataList_test.append({'i1': i1.unsqueeze(0), 'i2': i2})
        obsList_test.append(':- not document_type(0, i1, {}), not document_type_index(i2, {}).'.format(l, l))


    # define nnMapping and optimizers, initialze NeurASP object
    if use_dropout:
        m = Net_Dropout()
    else:
        m = Net()
    nnMapping = {'document_type': m}
    optimizers = {'document_type': torch.optim.Adam(m.parameters(), lr=learning_rate)}
    NeurASPobj = NeurASP(dprogram, nnMapping, optimizers)

    # generate name of file that holds the trained model
    model_file_name = "NeurASP_final_{}_{}_{}_{}_{}".format(seed, nb_epochs, batch_size, 
        learning_rate, use_dropout)

    # train and test
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