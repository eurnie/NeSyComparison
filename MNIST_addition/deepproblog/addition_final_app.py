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
from deepproblog.heuristics import geometric_mean, ucs
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.evaluate import get_confusion_matrix

sys.path.append("..")
from data.generate_dataset import generate_dataset_mnist, generate_dataset_fashion_mnist
from data.network_torch import Net

############################################### PARAMETERS ##############################################
method = "geometric_mean"
nb_epochs = 100
batch_size = 8
learning_rate = 0.001
dropout_rate = 0.2
loss_function_name = "cross_entropy"
size_val = 0.1
#########################################################################################################

# for dataset, label_noise in [("FashionMNIST", 0), ("MNIST", 0.1), ("MNIST", 0.25), ("MNIST", 0.5)]:
for dataset, label_noise in [("MNIST", 0)]:
    assert (dataset == "MNIST") or (dataset == "FashionMNIST")
    for seed in range(0, 10):
        # generate name of file that holds the trained model
        model_file_name = "DeepProbLog_final_{}_{}_{}_{}_{}_{}_{}".format(seed, 
            nb_epochs, size_val, dropout_rate, loss_function_name, learning_rate, batch_size)
        model_file_location = f'results/{method}/{dataset}/final/label_noise_{label_noise}/{model_file_name}'

        if not os.path.isfile(model_file_location):
            # setting seeds for reproducibility
            random.seed(seed)
            numpy.random.seed(seed)
            torch.manual_seed(seed)

            # generate and shuffle dataset
            if dataset == "MNIST":
                generate_dataset_mnist(seed, label_noise)
            elif dataset == "FashionMNIST":
                generate_dataset_fashion_mnist(seed, label_noise)

            # import train, val and test set
            train_set, val_set, test_set = import_datasets(dataset, size_val)

            network = Net(dropout_rate)
            net = Network(network, "mnist_net", batching=True)
            net.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

            model = Model("addition.pl", [net])
            if method == "exact":
                model.set_engine(ExactEngine(model), cache=True)
            elif method == "geometric_mean":
                model.set_engine(ApproximateEngine(model, 1, geometric_mean, exploration=False))

            if dataset == "MNIST":
                model.add_tensor_source("train", MNIST_train)
                model.add_tensor_source("val", MNIST_val)
                model.add_tensor_source("test", MNIST_test)
            elif dataset == "FashionMNIST":
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
                train = train_model(model, loader, 1, log_iter=100000, profile=0, loss_function_name=loss_function_name)
                total_training_time += time.time() - start_time
                val_accuracy = get_confusion_matrix(train.model, val_set).accuracy()
                print("Val accuracy after epoch", epoch, ":", val_accuracy)
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    nb_epochs_done = epoch + 1
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
            model.save_state(model_file_location)

            # testing
            start_time = time.time()
            accuracy = get_confusion_matrix(model, test_set).accuracy()
            testing_time = time.time() - start_time

            # save results to a summary file
            information = {
                "algorithm": "DeepProbLog",
                "seed": seed,
                "method": method,
                "nb_epochs": nb_epochs_done,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "loss_function": loss_function_name,
                "dropout_rate": dropout_rate,
                "size_val": size_val,
                "accuracy": accuracy,
                "training_time": total_training_time,
                "testing_time": testing_time,
                "model_files": model_file_name
            }
            with open(f'results/{method}/{dataset}/summary_final_{label_noise}.json', "a") as outfile:
                json.dump(information, outfile)
                outfile.write('\n')

            # print results
            print("############################################")
            print("Seed: {} \nAccuracy: {} \nTraining time: {} \nTesting time: {}".format(seed, accuracy, 
                total_training_time, testing_time))
            print("############################################")
