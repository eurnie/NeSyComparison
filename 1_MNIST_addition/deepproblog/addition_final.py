import time
import torch
import random
import numpy
import sys
from import_data import MNIST_train, MNIST_test, import_dataset
from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.evaluate import get_confusion_matrix

sys.path.append("..")
from data.generate_dataset import generate_dataset
from data.network_torch import Net, Net_Dropout

def train_and_test(train_set, test_set, method, max_nb_epochs, batch_size, learning_rate, use_dropout):
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

    model.add_tensor_source("train", MNIST_train)
    model.add_tensor_source("test", MNIST_test)
    loader = DataLoader(train_set, batch_size, False)

    # training
    start_time = time.time()
    train = train_model(model, loader, max_nb_epochs, log_iter=100000, profile=0)
    training_time = time.time() - start_time

    # testing
    accuracy = get_confusion_matrix(train.model, test_set).accuracy()

    return accuracy, training_time

############################################### PARAMETERS ##############################################
method = "exact"
nb_epochs = 1
batch_size = 2
learning_rate = 0.001
use_dropout = True
#########################################################################################################

for seed in range(0, 10):
    # setting seeds for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    # shuffle dataset
    generate_dataset(seed)

    # import train, val and test set
    train_set = import_dataset("train")
    # TODO: import val set
    test_set = import_dataset("test")

    # train and test
    accuracy, training_time = train_and_test(train_set, test_set, method, nb_epochs, batch_size, 
        learning_rate, use_dropout)

    # print results
    print("############################################")
    print("Seed: {} \nAccuracy: {} \nTraining time: {}".format(seed, accuracy, training_time))
    print("############################################")