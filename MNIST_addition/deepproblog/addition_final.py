import time
import torch
import random
import numpy
import sys
import os
import json
from import_data import *
from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.evaluate import get_confusion_matrix

sys.path.append("..")
from data.generate_dataset import generate_dataset_mnist, generate_dataset_fashion_mnist
from data.network_torch import Net, Net_Dropout

def train_and_test(dataset, model_file_name, train_set, val_set, test_set, method, nb_epochs, batch_size, 
        learning_rate, use_dropout):
    if use_dropout:
        network = Net_Dropout()
    else:
        network = Net()
    net = Network(network, "mnist_net", batching=True)
    net.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    model = Model("addition.pl", [net])
    if method == "exact":
        model.set_engine(ExactEngine(model), cache=True)
    elif method == "geometric_mean":
        model.set_engine(ApproximateEngine(model, 1, ApproximateEngine.geometric_mean, exploration=False))

    if dataset == "mnist":
        model.add_tensor_source("train", MNIST_train)
        model.add_tensor_source("val", MNIST_val)
        model.add_tensor_source("test", MNIST_test)
    elif dataset == "fashion_mnist":
        model.add_tensor_source("train", FashionMNIST_train)
        model.add_tensor_source("val", FashionMNIST_val)
        model.add_tensor_source("test", FashionMNIST_test)

    loader = DataLoader(train_set, batch_size, False)

    if not os.path.exists("best_model"):
        os.mkdir("best_model")

    # training (with early stopping)
    total_training_time = 0
    best_accuracy = 0
    counter = 0
    for epoch in range(nb_epochs):
        start_time = time.time()
        train = train_model(model, loader, 1, log_iter=100000, profile=0)
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
    model.save_state(f'results/{method}/{dataset}/{model_file_name}')

    # testing
    start_time = time.time()
    accuracy = get_confusion_matrix(model, test_set).accuracy()
    testing_time = time.time() - start_time

    return accuracy, total_training_time, testing_time

################################################# DATASET ###############################################
dataset = "mnist"
# dataset = "fashion_mnist"
label_noise = 0.1
#########################################################################################################

############################################### PARAMETERS ##############################################
method = "exact"
nb_epochs = 3
batch_size = 4
learning_rate = 0.001
use_dropout = False
size_val = 0.1
#########################################################################################################

for seed in range(0, 10):
    # setting seeds for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    # generate and shuffle dataset
    if dataset == "mnist":
        generate_dataset_mnist(seed, label_noise)
    elif dataset == "fashion_mnist":
        generate_dataset_fashion_mnist(seed, label_noise)

    # import train, val and test set
    train_set, val_set, test_set = import_datasets(dataset, size_val)

    # generate name of file that holds the trained model
    model_file_name = "final/label_noise_{}/DeepProbLog_final_{}_{}_{}_{}_{}_{}_{}".format(label_noise, seed, 
        method, nb_epochs, batch_size, learning_rate, use_dropout, size_val)

    # train and test
    accuracy, training_time, testing_time = train_and_test(dataset, model_file_name, train_set, val_set, 
        test_set, method, nb_epochs, batch_size, learning_rate, use_dropout)
    
    # save results to a summary file
    information = {
        "algorithm": "DeepProbLog",
        "seed": seed,
        "method": method,
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
    with open(f'results/{method}/{dataset}/final/label_noise_{label_noise}/summary_final.json', "a") as outfile:
        json.dump(information, outfile)
        outfile.write('\n')

    # print results
    print("############################################")
    print("Seed: {} \nAccuracy: {} \nTraining time: {} \nTesting time: {}".format(seed, accuracy, 
        training_time, testing_time))
    print("############################################")