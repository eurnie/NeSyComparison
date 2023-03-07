import torch
import random
import numpy
import sys
import json
from import_data import MNIST_train, MNIST_val, MNIST_test, import_datasets_kfold
from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.evaluate import get_confusion_matrix

sys.path.append("..")
from data.generate_dataset import generate_dataset
from data.network_torch import Net, Net_Dropout

def train_and_test(model_file_name_dir, train_set_list, method, nb_epochs, batch_size, learning_rate, 
    use_dropout):

    accuracies = []

    for fold_nb in range(1, 11):
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
            model.set_engine(ApproximateEngine(model, 1, ApproximateEngine.geometric_mean, 
                exploration=False))
        model.add_tensor_source("train", MNIST_train)
        model.add_tensor_source("val", MNIST_val)
        model.add_tensor_source("test", MNIST_test)

        for _ in range(nb_epochs):
            for i in range(0, 10):
                if (i != (fold_nb - 1)):
                    loader = DataLoader(train_set_list[i], batch_size, False)
                    train_model(model, loader, 1, log_iter=100000, profile=0)
                else:
                    test_set = train_set_list[i]

        # save trained model to a file
        model.save_state("results/param/{}/fold_{}".format(model_file_name_dir, fold_nb))

        # testing
        fold_accuracy = get_confusion_matrix(model, test_set).accuracy()
        accuracies.append(fold_accuracy)
        print(fold_nb, "-- Fold accuracy: ", fold_accuracy)

    return sum(accuracies) / 10, accuracies

############################################### PARAMETERS ##############################################
seed = 0
method = "exact"
nb_epochs = 3
batch_size = 2
learning_rate = 0.001
use_dropout = False
#########################################################################################################

# (1, 8, 0.001, True)
# (2, 4, 0.001, True)
# (2, 2, 0.001, False)
# (2, 2, 0.001, True)
# (3, 4, 0.001, True)
# (3, 8, 0.001, False)
# (1, 8, 0.001, False)

# setting seeds for reproducibility
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)

# shuffle dataset
generate_dataset(seed)

# import train set (already divided in 10 parts)
train_set_list = import_datasets_kfold()

# generate name of folder that holds all the trained models
model_file_name_dir = "DeepProbLog_param_{}_{}_{}_{}_{}_{}".format(seed, method, nb_epochs, 
    batch_size, learning_rate, use_dropout)

# train and test
avg_accuracy, accuracies = train_and_test(model_file_name_dir, train_set_list, method, nb_epochs, 
    batch_size, learning_rate, use_dropout)

# save results to a summary file
information = {
    "algorithm": "DeepProbLog",
    "seed": seed,
    "method": method,
    "nb_epochs": nb_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "use_dropout": use_dropout,
    "accuracies": accuracies,
    "avg_accuracy": avg_accuracy,
    "model_files_dir": model_file_name_dir
}
with open("results/summary_param.json", "a") as outfile:
    json.dump(information, outfile)
    outfile.write('\n')

# print results
print("############################################")
print("Seed: {} \nAvg_accuracy: {}".format(seed, avg_accuracy))
print("############################################")