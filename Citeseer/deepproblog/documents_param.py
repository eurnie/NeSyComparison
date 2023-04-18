import sys
import json
import random
import numpy
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

def train_and_test(model_file_name, train_set, val_set, method, nb_epochs, batch_size, 
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

    # training 
    train_model(model, loader, nb_epochs, log_iter=10, profile=0)

    # save trained model to a file
    model.save_state(f'results/{method}/param/{model_file_name}')

    # testing
    accuracy = get_confusion_matrix(model, val_set).accuracy()
    return accuracy

############################################### PARAMETERS ##############################################
seed = 0
method = "geometric_mean"
nb_epochs = 1
batch_size = 4
learning_rate = 0.001
use_dropout = False
#########################################################################################################

# setting seeds for reproducibility
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)

# import train, val and test set
train_set, val_set, _ = import_datasets()

# generate name of file that holds the trained model
model_file_name = "DeepProbLog_param_{}_{}_{}_{}_{}_{}".format(seed, method, nb_epochs, batch_size, 
    learning_rate, use_dropout)

# train and test
accuracy = train_and_test(model_file_name, train_set, val_set, 
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
    "model_file": model_file_name
}
with open(f'results/{method}/summary_param.json', "a") as outfile:
    json.dump(information, outfile)
    outfile.write('\n')

# print results
print("############################################")
print("Accuracy: {}".format(accuracy))
print("############################################")