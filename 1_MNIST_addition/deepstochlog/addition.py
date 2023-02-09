import torch
import random
import numpy

from pathlib import Path
from typing import Sequence, Union
from deepstochlog.context import Context, ContextualizedTerm
from torch.optim import Adam
from time import time
from deepstochlog.network import Network, NetworkStore
from import_data import get_mnist_data, parse_data
from network import Net
from deepstochlog.utils import create_model_accuracy_calculator, calculate_accuracy
from deepstochlog.dataloader import DataLoader
from deepstochlog.model import DeepStochLogModel
from deepstochlog.term import Term, List
from deepstochlog.trainer import DeepStochLogTrainer

############################################################################################
SEED_PYTHON = 123
SEED_NUMPY = 456
SEED_TORCH = 789
batch_size = 10
nb_epochs = 3
learning_rate = 1e-3
############################################################################################
log_iter = 1000
############################################################################################

# setting seeds for reproducibility
random.seed(SEED_PYTHON)
numpy.random.seed(SEED_NUMPY)
torch.manual_seed(SEED_TORCH)

root_path = Path(__file__).parent
t1 = Term("t1")
t2 = Term("t2")
argument_sequence = List(t1, t2)

class SimpleAdditionDataset(Sequence):
    def __init__(self, dataset_name, digit_length=1, size: int = None):
        self.mnist_dataset = get_mnist_data(dataset_name)
        self.ct_term_dataset = []
        # size = len(self.mnist_dataset) // 2
        if (dataset_name == "train"):
            dataset, labels = parse_data("../data/MNIST/processed/train.txt")
        elif (dataset_name == "test"):
            dataset, labels = parse_data("../data/MNIST/processed/test.txt")
        for idx in range(0, len(dataset)):
            mnist_datapoint_1 = self.mnist_dataset[dataset[idx][0][0]]
            mnist_datapoint_2 = self.mnist_dataset[dataset[idx][1][0]]
            digit_1 = mnist_datapoint_1[1]
            digit_2 = mnist_datapoint_2[1]
            total_sum = labels[idx]

            addition_term = ContextualizedTerm(
                # load context with the tensors
                context=Context({t1: mnist_datapoint_1[0], t2: mnist_datapoint_2[0]}),
                # create the term containing the sum and a list of tokens representing the tensors
                term=Term(
                    "addition",
                    Term(str(total_sum)),
                    argument_sequence,
                ),
                meta=str(digit_1) + "+" + str(digit_2),
            )
            self.ct_term_dataset.append(addition_term)

    def __len__(self):
        return len(self.ct_term_dataset)

    def __getitem__(self, item: Union[int, slice]):
        if type(item) is slice:
            return (self[i] for i in range(*item.indices(len(self))))
        return self.ct_term_dataset[item]

# create a network object, containing the MNIST network and the index list
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

model = DeepStochLogModel.from_file(file_location=str((root_path / "addition.pl").absolute()),
    query=query, networks=networks, device=device, verbose=False)
optimizer = Adam(model.get_all_net_parameters(), lr=learning_rate)
optimizer.zero_grad()

train_data = SimpleAdditionDataset("train")
test_data = SimpleAdditionDataset("test")

# own DataLoader that can deal with proof trees and tensors (replicates the pytorch dataloader interface)
train_dataloaders = []

for i in range(0, len(train_data), log_iter):
    train_data_iter = list(train_data[i:i+log_iter])
    train_dataloaders.append(DataLoader(train_data_iter, batch_size=batch_size))

test_dataloader = DataLoader(test_data, batch_size=1)

# create test functions
calculate_model_accuracy = create_model_accuracy_calculator(
    model, test_dataloader,  time(),
    val_dataloader=None,
)

trainer = DeepStochLogTrainer(log_freq=10000, accuracy_tester=calculate_model_accuracy)

total_training_time = 0
highest_accuracy = 0
highest_accuracy_index = 0

for epoch in range(0, nb_epochs):
    for nb in range(0, len(train_dataloaders)):
        loader = train_dataloaders[nb]
        total_training_time += trainer.train(model=model, optimizer=optimizer, dataloader=loader, epochs=1)

        accuracy = calculate_accuracy(model, test_dataloader)[0]

        if (accuracy > highest_accuracy):
            highest_accuracy = accuracy
            highest_accuracy_index = (epoch * 30000) + log_iter + (nb * log_iter)

        log_file = "results/results_deepstochlog_{}_{}_{}_{}_{}_{}.txt".format(SEED_PYTHON, SEED_NUMPY, SEED_TORCH, batch_size, nb_epochs, learning_rate)

        with open(log_file, "a") as f:
            f.write(str((epoch * 30000) + log_iter + (nb * log_iter)))
            f.write(" ")
            f.write(str(total_training_time))
            f.write(" ")
            f.write(str(accuracy))
            f.write(" ")
            f.write("\n")

        print("############################################")
        print("Number of entries: ", (epoch * 30000) + log_iter + (nb * log_iter))
        print("Total training time: ", total_training_time)
        print("Accuracy: ", accuracy)
        print("############################################")

print("The highest accuracy was {} and was reached (the first time) after seeing {} samples.".format(highest_accuracy, highest_accuracy_index))