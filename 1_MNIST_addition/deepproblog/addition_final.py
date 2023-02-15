import time
import torch
import random
import numpy
import sys
from import_data import MNIST_train, MNIST_val, MNIST_test, import_datasets
from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.evaluate import get_confusion_matrix

sys.path.append("..")
from data.generate_dataset import generate_dataset
from data.network_torch import Net, Net_Dropout

def train_and_test(train_set, val_set, test_set, method, nb_epochs, batch_size, learning_rate, use_dropout):
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
    model.add_tensor_source("val", MNIST_val)
    model.add_tensor_source("test", MNIST_test)
    loader = DataLoader(train_set, batch_size, False)

    # training (with early stopping)
    total_training_time = 0
    best_accuracy = 0
    counter = 0
    for epoch in range(nb_epochs):
        start_time = time.time()
        train = train_model(model, loader, 1, log_iter=100000, profile=0)
        total_training_time += time.time() - start_time
        val_accuracy = get_confusion_matrix(train.model, val_set).accuracy()
        print("Val accuracy after epoch", epoch, ":", val_accuracy)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            train.model.save_state("model/state", complete=True)
            counter = 0
        else:
            if counter >= 1:
                break
            counter += 1
    model.load_state("model/state")

    # testing
    start_time = time.time()
    accuracy = get_confusion_matrix(model, test_set).accuracy()
    testing_time = time.time() - start_time

    return accuracy, total_training_time, testing_time

############################################### PARAMETERS ##############################################
method = "exact"
nb_epochs = 3
batch_size = 2
learning_rate = 0.001
use_dropout = False
size_val = 0.1
#########################################################################################################

for seed in range(0, 10):
    # setting seeds for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    # shuffle dataset
    generate_dataset(seed)

    # import train, val and test set
    train_set, val_set, test_set = import_datasets(size_val)

    # train and test
    accuracy, training_time, testing_time = train_and_test(train_set, val_set, test_set, method, nb_epochs, 
        batch_size, learning_rate, use_dropout)

    # print results
    print("############################################")
    print("Seed: {} \nAccuracy: {} \nTraining time: {} \nTesting time: {}".format(seed, accuracy, 
        training_time, testing_time))
    print("############################################")