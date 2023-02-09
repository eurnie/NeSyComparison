import torch
import random
import numpy
import sys
import copy
from pathlib import Path
from torch.optim import Adam
from time import time
from itertools import product
from import_data import SimpleAdditionDataset
from deepstochlog.dataloader import DataLoader
from deepstochlog.model import DeepStochLogModel
from deepstochlog.term import Term, List
from deepstochlog.trainer import DeepStochLogTrainer   
from deepstochlog.network import Network, NetworkStore
from deepstochlog.utils import create_model_accuracy_calculator, calculate_accuracy, set_fixed_seed
from sklearn.model_selection import KFold

sys.path.append("..")
from data.generate_dataset import generate_dataset
from data.network_torch import Net

def train_and_test(train_set, test_set, max_nb_epochs, batch_size, learning_rate, epsilon):
    # create a network object containing the MNIST network and the index list
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
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

    # create test functions
    calculate_model_accuracy = create_model_accuracy_calculator(model, test_dataloader,  time(),
        val_dataloader=None)

    # training
    trainer = DeepStochLogTrainer(log_freq=100000, accuracy_tester=calculate_model_accuracy)
    training_time = trainer.train(model, optimizer, train_dataloader, max_nb_epochs, epsilon)

    # testing
    accuracy = calculate_accuracy(model, test_dataloader)[0]

    return accuracy, training_time

############################################### PARAMETERS ##############################################
possible_nb_epochs = [1, 2, 3]
possible_batch_size = [16, 32, 64]
possible_learning_rate = [0.001]
possible_epsilon = [0.00000001]
k = 10
#########################################################################################################

for seed in range(0, 10):
    for param in product(possible_nb_epochs, possible_batch_size, possible_learning_rate, possible_epsilon):
        nb_epochs, batch_size, learning_rate, epsilon = param

        # setting seeds for reproducibility
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)

        # extra seeds?
        # set_import_data_seed(seed)
        # set_fixed_seed(seed)
        # torch.use_deterministic_algorithms(True)
        # torch.backends.cudnn.benchmark = False

        # set the computation mode to be deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # import train and test set (shuffled)
        # generate_dataset(seed)
        train_set = SimpleAdditionDataset("train")
        
        k_fold_set = []
        for i in range(0, len(train_set), int(len(train_set) / k)):
            fold = list(train_set[i:int(i+(len(train_set)/k))])
            k_fold_set.append(fold)

        accuracy = 0

        for i, val_set in enumerate(k_fold_set):
            started = False
            for j, split in enumerate(k_fold_set):
                if i != j:
                    if not started:
                        total_dataset = copy.deepcopy(split)
                        started = True
                    else:
                        total_dataset += split
            assert len(total_dataset) == 27000
            assert len(val_set) == 3000
            fold_accuracy, _ = train_and_test(total_dataset, val_set, nb_epochs, batch_size, learning_rate, epsilon)
            print("Fold accuracy: ", fold_accuracy)
            accuracy += fold_accuracy

        accuracy /= k

        # print results
        print("############################################")
        print("seed: {}".format(seed))
        print("nb_epochs: {}".format(nb_epochs))
        print("batch_size: {}".format(batch_size))
        print("learning_rate: {}".format(learning_rate))
        print("epsilon: {}".format(epsilon))
        print("----------")
        print("Accuracy: {}".format(accuracy))
        print("############################################")