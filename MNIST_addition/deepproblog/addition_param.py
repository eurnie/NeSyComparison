import torch
import random
import numpy
import sys
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
from data.network_torch import Net, Net_Dropout

################################################# DATASET ###############################################
dataset = "mnist"
# dataset = "fashion_mnist"
#########################################################################################################

############################################### PARAMETERS ##############################################
seed = 0
method = "exact"
nb_epochs = 100
batch_size = 2
learning_rate = 0.001
use_dropout = False
size_val = 0.1
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

# import train and val set
train_set, val_set, _ = import_datasets(dataset, size_val)

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
    model.set_engine(ApproximateEngine(model, 1, geometric_mean, exploration=True))
    
if dataset == "mnist":
    model.add_tensor_source("train", MNIST_train)
    model.add_tensor_source("val", MNIST_val)
    model.add_tensor_source("test", MNIST_test)
elif dataset == "fashion_mnist":
    model.add_tensor_source("train", FashionMNIST_train)
    model.add_tensor_source("val", FashionMNIST_val)
    model.add_tensor_source("test", FashionMNIST_test)

loader = DataLoader(train_set, batch_size, False)

best_accuracy = 0

# training (no early stopping)
for epoch in range(nb_epochs):
    train_model(model, loader, 1, log_iter=100, profile=0)

    # generate name of folder that holds all the trained models
    model_file_name_dir = "DeepProbLog_param_{}_{}_{}_{}_{}_{}_{}".format(seed, method, epoch + 1, 
        batch_size, learning_rate, use_dropout, size_val)
    
    # save trained model to a file
    model.save_state(f'results/{method}/{dataset}/param/{model_file_name_dir}')

    # testing
    accuracy = get_confusion_matrix(model, val_set).accuracy()

    # save results to a summary file
    information = {
        "algorithm": "DeepProbLog",
        "seed": seed,
        "method": method,
        "nb_epochs": epoch + 1,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "use_dropout": use_dropout,
        "size_val": size_val,
        "accuracy": accuracy,
        "model_files_dir": model_file_name_dir
    }
    with open(f'results/{method}/{dataset}/param/summary_param.json', "a") as outfile:
        json.dump(information, outfile)
        outfile.write('\n')

    # print results
    print("############################################")
    print("Accuracy: {}".format(accuracy))
    print("############################################")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        counter = 0
    else:
        if counter >= 3:
            break
        counter += 1