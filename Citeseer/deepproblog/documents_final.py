import sys
import os
import json
import random
import numpy
import time
import torch
from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.heuristics import geometric_mean, ucs
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.evaluate import get_confusion_matrix
from import_data import import_datasets, citeseer_examples

sys.path.append("..")
from data.network_torch import Net, Net_Dropout

def train_and_test(model_file_name, train_set, val_set, test_set, method, nb_epochs, batch_size, 
        learning_rate, use_dropout):
    if use_dropout:
        network = Net_Dropout()
    else:
        network = Net()

    net = Network(network, "citeseer_net", batching=True)
    net.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    model = Model("documents.pl", [net])
    if method == "exact":
        model.set_engine(ExactEngine(model), cache=False)
    elif method == "geometric_mean":
        model.set_engine(ApproximateEngine(model, 1, geometric_mean, exploration=False))   

    model.add_tensor_source("citeseer", citeseer_examples)
    loader = DataLoader(train_set, batch_size, False)

    if not os.path.exists("best_model"):
        os.mkdir("best_model")

    # training (with early stopping)
    total_training_time = 0
    best_accuracy = 0
    counter = 0
    for epoch in range(nb_epochs):
        start_time = time.time()
        train = train_model(model, loader, 1, log_iter=10, profile=0)
        total_training_time += time.time() - start_time
        val_accuracy = get_confusion_matrix(train.model, val_set).accuracy()
        print("Val accuracy after epoch", epoch, ":", val_accuracy)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            train.model.save_state("best_model/state", complete=True)
            counter = 0
        else:
            if counter >= 1:
                break
            counter += 1

    # early stopping: load best model and delete file
    model.load_state("best_model/state")
    os.remove("best_model/state")
    os.rmdir("best_model")

    # save trained model to a file
    model.save_state(f'results/{method}/final/{model_file_name}')

    # testing
    start_time = time.time()
    accuracy = get_confusion_matrix(model, test_set).accuracy()
    testing_time = time.time() - start_time

    return accuracy, total_training_time, testing_time

############################################### PARAMETERS ##############################################
method = "exact"
nb_epochs = 1
batch_size = 8
learning_rate = 0.001
use_dropout = False
#########################################################################################################

for seed in range(0, 10):
    # setting seeds for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    # import train, val and test set
    train_set, val_set, test_set = import_datasets()

    # generate name of file that holds the trained model
    model_file_name = "DeepProbLog_final_{}_{}_{}_{}_{}_{}".format(seed, method, nb_epochs, batch_size, 
        learning_rate, use_dropout)

    # train and test
    accuracy, training_time, testing_time = train_and_test(model_file_name, train_set, val_set, test_set, 
        method, nb_epochs, batch_size, learning_rate, use_dropout)
    
    # save results to a summary file
    information = {
        "algorithm": "DeepProbLog",
        "seed": seed,
        "method": method,
        "nb_epochs": nb_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "use_dropout": use_dropout,
        "accuracy": accuracy,
        "training_time": training_time,
        "testing_time": testing_time,
        "model_file": model_file_name
    }
    with open(f'results/{method}/summary_final.json', "a") as outfile:
        json.dump(information, outfile)
        outfile.write('\n')

    # print results
    print("############################################")
    print("Seed: {} \nAccuracy: {} \nTraining time: {} \nTesting time: {}".format(seed, accuracy, 
        training_time, testing_time))
    print("############################################")