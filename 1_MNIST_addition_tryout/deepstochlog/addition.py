import sys
sys.path.append('../')
import torch
from time import time
from torch.optim import Adam
from pathlib import Path
from typing import Sequence, Union
from data.import_data import addition, datasets
from network import AdditionNet
from deepstochlog.context import Context, ContextualizedTerm
from deepstochlog.network import Network, NetworkStore
from deepstochlog.utils import (
    create_run_test_query,
    create_model_accuracy_calculator,
    calculate_accuracy
)
from deepstochlog.dataloader import DataLoader
from deepstochlog.model import DeepStochLogModel
from deepstochlog.term import Term, List
from deepstochlog.trainer import DeepStochLogTrainer, print_logger

####################
# hyperparameters
####################

epochs = 10
batch_size = 1
learning_rate = 1e-3
epsilon = 1e-8

####################
# setup
####################

root_path = Path(__file__).parent

t1 = Term("t1")
t2 = Term("t2")
argument_sequence = List(t1, t2)

class SimpleAdditionDataset(Sequence):
    def __init__(self, dataset: str, digit_length=1):
        self.mnist_dataset = addition(1, dataset)
        self.ct_term_dataset = []
        for index in range(len(self.mnist_dataset)):
        # for index in range(100):
            if (index % 1000 == 0):
                print(index)
            digit_1 = self.mnist_dataset[index][0]
            digit_2 = self.mnist_dataset[index][1]
            label = self.mnist_dataset[index][2]

            addition_term = ContextualizedTerm(
                # Load context with the tensors
                context=Context({t1: digit_1[0], t2: digit_2[0]}),
                # Create the term containing the sum and a list of tokens representing the tensors
                term=Term(
                    "addition",
                    Term(str(label)),
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
mnist_classifier = Network(
    "number", 
    AdditionNet(), 
    index_list=[Term(str(i)) for i in range(10)]
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
networks = NetworkStore(mnist_classifier)

# Load the model "addition_simple.pl" with the specific query
query = Term(
    # We want to use the "addition" non-terminal in the specific grammar
    "addition",
    # We want to calculate all possible sums by giving the wildcard "_" as argument
    Term("_"),
    # Denote that the input will be a list of two tensors, t1 and t2, representing the MNIST digit.
    List(Term("t1"), Term("t2")),
)

model = DeepStochLogModel.from_file(
    file_location=str((root_path / "addition.pl").absolute()),
    query=query,
    networks=networks,
    device=device,
    verbose=True,
)

optimizer = Adam(model.get_all_net_parameters(), lr=learning_rate)
optimizer.zero_grad()

print("Creating training dataset... takes a while")

training_set = SimpleAdditionDataset(
    digit_length=1,
    dataset="train"
)

print("Creating testing dataset... takes a while")

testing_set = SimpleAdditionDataset(
    digit_length=1,
    dataset="test"
)

train_dataloader = DataLoader(training_set, batch_size=batch_size)
test_dataloader = DataLoader(testing_set, batch_size=batch_size)
test_dataloader_new = DataLoader([testing_set[0]], batch_size=batch_size)

####################
# training and testing
####################

start_time = time()

# Create test functions
run_test_query = create_run_test_query(
    model=model,
    test_data=[testing_set[0]],
    test_example_idx=None,
    verbose=True,
)
calculate_model_accuracy = create_model_accuracy_calculator(
    model,
    test_dataloader_new,
    start_time,
    val_dataloader=test_dataloader_new,
)

# Train the DeepStochLog model
trainer = DeepStochLogTrainer(
    log_freq=100,
    accuracy_tester=calculate_model_accuracy,
    logger=print_logger,
    print_time=False,
    test_query=None,
)

for i in range(epochs):
    print("\nEpoch {}\n-------------------------------".format(i+1))

    trainer.train(
        model=model,
        optimizer=optimizer,
        dataloader=train_dataloader,
        epochs=1,
        epsilon=epsilon
    )

    print("ACCURACY: ", 100 * calculate_accuracy(model, test_dataloader)[0])