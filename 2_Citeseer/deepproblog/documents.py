import time
import torch
import random
import numpy
import sys
from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.evaluate import get_confusion_matrix
from import_data import import_dataset, Citeseer_train, Citeseer_test

sys.path.append("..")
from data.network_torch import Net

def train_and_test(train_set, test_set, method, nb_epochs, learning_rate):
    network = Net()
    net = Network(network, "citeseer_net", batching=False)
    net.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    model = Model("documents.pl", [net])
    if method == "exact":
        model.set_engine(ExactEngine(model), cache=True)
    elif method == "geometric_mean":
        model.set_engine(ApproximateEngine(model, 1, ApproximateEngine.geometric_mean, exploration=False))

    model.add_tensor_source("train", Citeseer_train)
    model.add_tensor_source("test", Citeseer_test)
    loader = DataLoader(train_set, 1, False)

    # training
    start_time = time.time()
    train = train_model(model, loader, nb_epochs, log_iter=100000, profile=0)
    training_time = time.time() - start_time

    # testing
    accuracy = get_confusion_matrix(train.model, test_set).accuracy()

    return accuracy, training_time

############################################### PARAMETERS ##############################################
method = "exact"
nb_epochs = 3
learning_rate = 0.001
#########################################################################################################

for seed in range(0, 10):
    # setting seeds for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    # import train and test set
    train_set = import_dataset("train")
    validation_set = import_dataset("val")
    test_set = import_dataset("test")

    # train and test the method on the MNIST addition dataset
    accuracy, training_time = train_and_test(train_set, test_set, method, nb_epochs, learning_rate)
    
    # print results
    print("############################################")
    print("Seed: {} \nAccuracy: {} \nTraining time: {}".format(seed, accuracy, training_time))
    print("############################################")