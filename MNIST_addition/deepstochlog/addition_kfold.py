import torch
import random
import numpy
import sys
import os
import time
import pickle
import json
from pathlib import Path
from torch.optim import Adam
from import_data import import_datasets_kfold
from deepstochlog.dataloader import DataLoader
from deepstochlog.model import DeepStochLogModel
from deepstochlog.term import Term, List
from deepstochlog.trainer import DeepStochLogTrainer
from deepstochlog.network import Network, NetworkStore
from deepstochlog.utils import create_model_accuracy_calculator, calculate_accuracy

sys.path.append("..")
from data.generate_dataset import generate_dataset_mnist, generate_dataset_fashion_mnist
from data.network_torch import Net

def train_and_test(dataset, model_file_name_dir, train_set_list, nb_epochs, batch_size, learning_rate, 
    epsilon, dropout_rate):

    accuracies = []

    for fold_nb in range(1, 11):
        # create a network object containing the MNIST network and the index list
        mnist_classifier = Network("number", Net(dropout_rate), index_list=[Term(str(i)) for i in range(10)])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        networks = NetworkStore(mnist_classifier)

        # load the model "addition_simple.pl" with the specific query
        query = Term(
            # we want to use the "addition" non-terminal in the specific grammar
            "addition",
            # we want to calculate all possible sums by giving the wildcard "_" as argument
            Term("_"),
            # denote that the input will be a list of two tensors, t1 and t2, representing the MNIST digit.
            List(Term("t1"), Term("t2")),
        )

        root_path = Path(__file__).parent
        model = DeepStochLogModel.from_file(file_location=str((root_path / "addition.pl").absolute()),
            query=query, networks=networks, device=device, verbose=False)
        optimizer = Adam(model.get_all_net_parameters(), lr=learning_rate)
        optimizer.zero_grad()     
        
        dummy_dataloader = DataLoader([], batch_size=1, shuffle=False)

        # create test functions
        calculate_model_accuracy = create_model_accuracy_calculator(model, dummy_dataloader,  time.time())

        # training
        trainer = DeepStochLogTrainer(log_freq=100000, accuracy_tester=calculate_model_accuracy)
        for epoch in range(nb_epochs):
            for i in range(0, 10):
                if (i != (fold_nb - 1)):
                    train_dataloader = DataLoader(train_set_list[i], batch_size=batch_size, shuffle=False)
                    trainer.train(model, optimizer, train_dataloader, 1, epsilon)
                else:
                    test_dataloader = DataLoader(train_set_list[i], batch_size=1, shuffle=False)
            print("Epoch", epoch + 1, "finished.")

        # save trained model to a file
        path = f'results/{dataset}/kfold/{model_file_name_dir}'
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f'results/{dataset}/kfold/{model_file_name_dir}/fold_{fold_nb}', "wb+") as handle:
            pickle.dump(model.neural_networks, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # testing
        fold_accuracy = calculate_accuracy(model, test_dataloader)[0]
        accuracies.append(fold_accuracy)
        print(fold_nb, "-- Fold accuracy: ", fold_accuracy)

    return sum(accuracies) / 10, accuracies

################################################# DATASET ###############################################
dataset = "mnist"
# dataset = "fashion_mnist"
#########################################################################################################

############################################### PARAMETERS ##############################################
seed = 0
nb_epochs = 1
batch_size = 512
learning_rate = 0.001
epsilon = 0.00000001
dropout_rate = 0
#########################################################################################################

# setting seeds for reproducibility
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)

# generate and shuffle dataset
if dataset == "mnist":
    generate_dataset_mnist(seed, 0)
elif dataset == "fashion_mnist":
    generate_dataset_fashion_mnist(seed, 0)

# import train set
train_set_list = import_datasets_kfold(dataset)

# generate name of folder that holds all the trained models
model_file_name_dir = "DeepStochLog_kfold_{}_{}_{}_{}_{}_{}".format(seed, nb_epochs, batch_size, 
    learning_rate, epsilon, dropout_rate)

# train and test
avg_accuracy, accuracies = train_and_test(dataset, model_file_name_dir, train_set_list, nb_epochs, 
                                          batch_size, learning_rate, epsilon, dropout_rate)

# save results to a summary file
information = {
    "algorithm": "DeepStochLog",
    "seed": seed,
    "nb_epochs": nb_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "epsilon": epsilon,
    "dropout_rate": dropout_rate,
    "accuracies": accuracies,
    "avg_accuracy": avg_accuracy,
    "model_files_dir": model_file_name_dir
}
with open(f'results/{dataset}/kfold/summary_kfold.json', "a") as outfile:
    json.dump(information, outfile)
    outfile.write('\n')

# print results
print("############################################")
print("Seed: {} \nAvg_accuracy: {}".format(seed, avg_accuracy))
print("############################################")