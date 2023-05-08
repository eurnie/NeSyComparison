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
from data.network_torch import Net_CiteSeer, Net_Cora, Net_PubMed

def train_and_test(dataset, model_file_name, train_set, val_set, test_set, method, nb_epochs, batch_size, 
        learning_rate, dropout_rate):
    if dataset == "CiteSeer":
        network = Net_CiteSeer(dropout_rate)
    elif dataset == "Cora":
        network = Net_CiteSeer(dropout_rate)
    elif dataset == "PubMed":
        network = Net_CiteSeer(dropout_rate)

    net = Network(network, "document_net", batching=True)
    net.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    model = Model(f'documents_{dataset}.pl', [net])
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
            if counter >= 2:
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

################################################# DATASET ###############################################
dataset = "CiteSeer"
move_to_test_set_ratio = 0
#########################################################################################################

############################################### PARAMETERS ##############################################
method = "exact"
nb_epochs = 100
batch_size = 64
learning_rate = 0.001
dropout_rate = 0
#########################################################################################################

for seed in range(0, 10):
    # setting seeds for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    # import train, val and test set
    train_set, val_set, test_set = import_datasets(move_to_test_set_ratio, seed)

    # generate name of file that holds the trained model
    model_file_name = "DeepProbLog_final_{}_{}_{}_{}_{}_{}_{}".format(seed, method, nb_epochs, 
        batch_size, learning_rate, dropout_rate, move_to_test_set_ratio)

    # train and test
    accuracy, training_time, testing_time = train_and_test(dataset, model_file_name, train_set, 
        val_set, test_set, method, nb_epochs, batch_size, learning_rate, dropout_rate)
    
    # save results to a summary file
    information = {
        "algorithm": "DeepProbLog",
        "seed": seed,
        "method": method,
        "nb_epochs": nb_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dropout_rate": dropout_rate,
        "accuracy": accuracy,
        "training_time": training_time,
        "testing_time": testing_time,
        "model_file": model_file_name
    }
    with open(f'results/{method}/{dataset}/summary_final_{move_to_test_set_ratio}.json', "a") as outfile:
        json.dump(information, outfile)
        outfile.write('\n')

    # print results
    print("############################################")
    print("Seed: {} \nAccuracy: {} \nTraining time: {} \nTesting time: {}".format(seed, accuracy, 
        training_time, testing_time))
    print("############################################")