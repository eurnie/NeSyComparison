import torch
import random
import numpy
import sys
import time
import pickle
from pathlib import Path
from torch.optim import Adam
from import_data import import_datasets
from deepstochlog.dataloader import DataLoader
from deepstochlog.model import DeepStochLogModel
from deepstochlog.term import Term, List
from deepstochlog.trainer import DeepStochLogTrainer
from deepstochlog.network import Network, NetworkStore
from deepstochlog.utils import create_model_accuracy_calculator, calculate_accuracy, set_fixed_seed

sys.path.append("..")
from data.generate_dataset import generate_dataset
from data.network_torch import Net, Net_Dropout

def train_and_test(train_set, val_set, test_set, nb_epochs, batch_size, learning_rate, epsilon, use_dropout):
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
    # if shuffle is set to True, also give the seed: the seed of the random package had only effect on this file
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)
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
            with open("best_model.pickle", "wb") as handle:
                pickle.dump(model.neural_networks, handle, protocol=pickle.HIGHEST_PROTOCOL)
            counter = 0
        else:
            if counter >= 1:
                break
            counter += 1
    with open("best_model.pickle", "rb") as handle:
        neural_networks = pickle.load(handle)
    model.neural_networks = neural_networks

    # testing
    start_time = time.time()
    accuracy = calculate_accuracy(model, test_dataloader)[0]
    testing_time = time.time() - start_time

    return accuracy, total_training_time, testing_time

############################################### PARAMETERS ##############################################
nb_epochs = 1
batch_size = 2
learning_rate = 0.001
epsilon = 0.00000001
use_dropout = False
size_val = 0.1
#########################################################################################################

for seed in range(0, 10):
    # setting seeds for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    # TODO: find out why method is not deterministic
    set_fixed_seed(seed)
    torch.cuda.empty_cache()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # shuffle dataset
    generate_dataset(seed)

    # import train, val and test set
    train_set, val_set, test_set = import_datasets(size_val)

    # train and test
    accuracy, training_time, testing_time = train_and_test(train_set, val_set, test_set, nb_epochs, 
        batch_size, learning_rate, epsilon, use_dropout)

    # print results
    print("############################################")
    print("Seed: {} \nAccuracy: {} \nTraining time: {} \nTesting time: {}".format(seed, accuracy, 
        training_time, testing_time))
    print("############################################")