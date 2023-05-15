import os
import torch
import random
import numpy
import sys
import time
import pickle
import json
from pathlib import Path
from torch.optim import Adam
from import_data import import_datasets
from deepstochlog.dataloader import DataLoader
from deepstochlog.model import DeepStochLogModel
from deepstochlog.term import Term, List
from deepstochlog.trainer import DeepStochLogTrainer
from deepstochlog.network import Network, NetworkStore
from deepstochlog.utils import create_model_accuracy_calculator, calculate_accuracy

sys.path.append("..")
from data.generate_dataset import generate_dataset_mnist, generate_dataset_fashion_mnist
from data.network_torch import Net

################################################# DATASET ###############################################
dataset = "MNIST"
#########################################################################################################

############################################### PARAMETERS ##############################################
seed = 0
nb_epochs = 100
size_val = 0.1
#########################################################################################################

for dropout_rate in [0, 0.2]:
    for epsilon in [0.00000001, 0.0000001]:
        for learning_rate in [0.001, 0.0001]:
            for batch_size in [2, 8, 32, 128]:
                # generate name of file that holds the trained model
                model_file_name = "DeepStochLog_param_{}_{}_{}_{}_{}_{}_{}".format(seed, 
                    nb_epochs, batch_size, learning_rate, epsilon, dropout_rate, size_val)
                model_file_location = f'results/{dataset}/param/{model_file_name}'

                if not os.path.isfile(model_file_location):
                    # setting seeds for reproducibility
                    random.seed(seed)
                    numpy.random.seed(seed)
                    torch.manual_seed(seed)

                    # generate and shuffle dataset
                    if dataset == "MNIST":
                        generate_dataset_mnist(seed, 0)
                    elif dataset == "FashionMNIST":
                        generate_dataset_fashion_mnist(seed, 0)

                    # import train and val set
                    train_set, val_set, _ = import_datasets(dataset, size_val)

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

                    # DataLoader that can deal with proof trees and tensors (replicates the pytorch dataloader interface)
                    # if shuffle is set to True, also give the seed: the seed of the random package had only effect on this 
                    # file
                    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
                    val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False)
                    dummy_dataloader = DataLoader([], batch_size=1, shuffle=False)

                    # create test functions
                    calculate_model_accuracy = create_model_accuracy_calculator(model, dummy_dataloader,  time.time())

                    # training (with early stopping)
                    trainer = DeepStochLogTrainer(log_freq=100000, accuracy_tester=calculate_model_accuracy)
                    total_training_time = 0
                    best_accuracy = 0
                    counter = 0
                    for epoch in range(nb_epochs):
                        total_training_time += trainer.train(model, optimizer, train_dataloader, 1, epsilon)
                        val_accuracy = calculate_accuracy(model, val_dataloader)[0]
                        print("Val accuracy after epoch", epoch, ":", val_accuracy)
                        if val_accuracy > best_accuracy:
                            best_accuracy = val_accuracy
                            nb_epochs_done = epoch + 1
                            with open("best_model.pickle", "wb") as handle:
                                pickle.dump(model.neural_networks, handle, protocol=pickle.HIGHEST_PROTOCOL)
                            counter = 0
                        else:
                            if counter >= 2:
                                break
                            counter += 1

                    with open("best_model.pickle", "rb") as handle:
                        neural_networks = pickle.load(handle)
                    model.neural_networks = neural_networks

                    os.remove("best_model.pickle")

                    # save trained model to a file
                    with open(model_file_location, "wb") as handle:
                        pickle.dump(model.neural_networks, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    # save results to a summary file
                    information = {
                        "algorithm": "DeepStochLog",
                        "seed": seed,
                        "nb_epochs": nb_epochs_done,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "epsilon": epsilon,
                        "dropout_rate": dropout_rate,
                        "size_val": size_val,
                        "accuracy": best_accuracy,
                        "model_file": model_file_name
                    }
                    with open(f'results/{dataset}/summary_param.json', "a") as outfile:
                        json.dump(information, outfile)
                        outfile.write('\n')

                    # print results
                    print("############################################")
                    print("Accuracy: {}".format(best_accuracy))
                    print("############################################")