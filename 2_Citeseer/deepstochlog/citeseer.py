from pathlib import Path
from typing import Union
from shutil import copy2
import torch
from torch.optim import Adam
from time import time

from deepstochlog.network import Network, NetworkStore
from citeseer_data import train_dataset, valid_dataset, test_dataset, queries_for_model, citations
from citeseer_utils import AccuracyCalculator
from models import Classifier
from deepstochlog.utils import set_fixed_seed
from deepstochlog.dataloader import DataLoader
from deepstochlog.model import DeepStochLogModel
from deepstochlog.trainer import DeepStochLogTrainer, print_logger, PrintFileLogger
from deepstochlog.term import Term, List

root_path = Path(__file__).parent

class GreedyDumbEvaluation:
    def __init__(self, documents, labels, store):
        self.store = store
        self.documents = documents
        self.labels = labels

    def __call__(self):
        classifier = self.store.networks["classifier"].neural_model
        classifier.eval()

        acc = torch.mean((torch.argmax(classifier(self.documents),dim=1) == self.labels).float())

        classifier.train()
        return "%s" % str(acc.numpy().tolist())

epochs=100
batch_size=32
lr=0.01
allow_division=True
device_str: str = None
train_size=None
test_size=None
logger=print_logger
test_batch_size=100
verbose=True
test_example_idx=0
expression_max_length=1
expression_length=1
seed=0
log_freq=1
set_fixed_seed(seed)

# Load the MNIST model, and Adam optimiser
input_size = len(train_dataset.documents[0])
classifier = Classifier(input_size=input_size)
classifier_network = Network(
    "classifier",
    classifier,
    index_list=[Term(str(op)) for op in range(6)],
)
networks = NetworkStore(classifier_network)

if device_str is not None:
    device = torch.device(device_str)
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


proving_start = time()
model = DeepStochLogModel.from_file(
    file_location=str((root_path / "citeseer.pl").absolute()),
    query=queries_for_model,
    networks=networks,
    device=device,
    prolog_facts= citations,
    verbose=verbose
)
proving_time = time() - proving_start

optimizer = Adam(model.get_all_net_parameters(), lr=lr)
optimizer.zero_grad()

if verbose:
    logger.print("\nProving the program took {:.2f} seconds".format(proving_time))



# Own DataLoader that can deal with proof trees and tensors (replicates the pytorch dataloader interface)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

# Create test functions
# run_test_query = create_run_test_query_probability(
#     model, test_data, test_example_idx, verbose
# )
calculate_model_accuracy = AccuracyCalculator( model=model,
                                                valid=valid_dataset,
                                                test=test_dataset,
                                                start_time=time())
# run_test_query = create_run_test_query(model, test_data, test_example_idx, verbose)
# calculate_model_accuracy = create_model_accuracy_calculator(model, test_dataloader, start_time)
# g = GreedyEvaluation(valid_data, test_data, networks)
# calculate_model_accuracy = "Acc", GreedyEvaluation(documents, labels, networks)

# Train the DeepStochLog model
trainer = DeepStochLogTrainer(
    log_freq=log_freq,
    accuracy_tester=(calculate_model_accuracy.header, calculate_model_accuracy),
    logger=logger,
    print_time=verbose,
)
trainer.train(
    model=model,
    optimizer=optimizer,
    dataloader=train_dataloader,
    epochs=epochs,
)