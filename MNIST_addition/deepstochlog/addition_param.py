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
from data.network_torch import Net, Net_Dropout

################################################# DATASET ###############################################
dataset = "mnist"
# dataset = "fashion_mnist"
#########################################################################################################

############################################### PARAMETERS ##############################################
seed = 0
nb_epochs = 100
batch_size = 2
learning_rate = 0.001
epsilon = 0.00000001
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

# import train, val and test set
train_set, val_set, _ = import_datasets(dataset, size_val)

# create a network object containing the MNIST network and the index list
if use_dropout:
    mnist_classifier = Network("number", Net_Dropout(), index_list=[Term(str(i)) for i in range(10)])
else:
    mnist_classifier = Network("number", Net(), index_list=[Term(str(i)) for i in range(10)])
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
trainer = DeepStochLogTrainer(log_freq=100, accuracy_tester=calculate_model_accuracy)

best_accuracy = 0

for epoch in range(nb_epochs):
    trainer.train(model, optimizer, train_dataloader, 1, epsilon)

    # generate name of file that holds the trained model
    model_file_name = "param/DeepStochLog_param_{}_{}_{}_{}_{}_{}_{}".format(seed, epoch + 1, batch_size, 
        learning_rate, epsilon, use_dropout, size_val)

    # save trained model to a file
    with open(f'results/{dataset}/{model_file_name}', "wb") as handle:
        pickle.dump(model.neural_networks, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # testing
    accuracy = calculate_accuracy(model, val_dataloader)[0]

    # save results to a summary file
    information = {
        "algorithm": "DeepStochLog",
        "seed": seed,
        "nb_epochs": epoch + 1,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epsilon": epsilon,
        "use_dropout": use_dropout,
        "size_val": size_val,
        "accuracy": accuracy,
        "model_file": model_file_name
    }
    with open(f'results/{dataset}/param/summary_param.json', "a") as outfile:
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
        if counter >= 2:
            break
        counter += 1