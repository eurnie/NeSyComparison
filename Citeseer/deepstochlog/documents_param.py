import random
import numpy
import json
import time
import pickle
import torch
import sys
from pathlib import Path
from torch.optim import Adam
from deepstochlog.network import Network, NetworkStore
from import_data import get_dataset, citations
from deepstochlog.dataloader import DataLoader
from deepstochlog.model import DeepStochLogModel
from deepstochlog.trainer import DeepStochLogTrainer, print_logger
from deepstochlog.term import Term
from citeseer_utils import create_model_accuracy_calculator, calculate_accuracy

sys.path.append("..")
from data.network_torch import Net_CiteSeer, Net_Cora, Net_PubMed

################################################# DATASET ###############################################
dataset = "CiteSeer"
move_to_test_set_ratio = 0
#########################################################################################################

############################################### PARAMETERS ##############################################
seed = 0
nb_epochs = 100
batch_size = 64
learning_rate = 0.001
epsilon = 0.00000001
dropout_rate = 0
size_val = 0.1
#########################################################################################################

assert (dataset == "CiteSeer") or (dataset == "Cora") or (dataset == "PubMed")
logger = print_logger

# setting seeds for reproducibility
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)

train_dataset, queries_for_model = get_dataset("train", seed)
val_dataset, _ = get_dataset("val", seed)

if dataset == "CiteSeer":
    classifier_network = Network("classifier", Net_CiteSeer(dropout_rate), index_list=[Term(str(op)) for op in range(6)])
elif dataset == "Cora":
    classifier_network = Network("classifier", Net_Cora(dropout_rate), index_list=[Term(str(op)) for op in range(7)])
elif dataset == "PubMed":
    classifier_network = Network("classifier", Net_PubMed(dropout_rate), index_list=[Term(str(op)) for op in range(3)])

networks = NetworkStore(classifier_network)

proving_start = time.time()
root_path = Path(__file__).parent
model = DeepStochLogModel.from_file(
    file_location=str((root_path / "documents.pl").absolute()),
    query=queries_for_model,
    networks=networks,
    prolog_facts=citations,
    verbose=False
)
proving_time = time.time() - proving_start

optimizer = Adam(model.get_all_net_parameters(), lr=learning_rate)
optimizer.zero_grad()
logger.print("\nProving the program took {:.2f} seconds".format(proving_time))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
dummy_dataloader = DataLoader([], batch_size=1, shuffle=False)

# create test functions
calculate_model_accuracy = create_model_accuracy_calculator(model, dummy_dataloader,  time.time())

# training
trainer = DeepStochLogTrainer(log_freq=100, accuracy_tester=calculate_model_accuracy)
best_accuracy = 0

for epoch in range(nb_epochs):
    trainer.train(model, optimizer, train_dataloader, 1, epsilon)

    # generate name of file that holds the trained model
    model_file_name = "DeepStochLog_param_{}_{}_{}_{}_{}_{}".format(seed, epoch + 1, 
        batch_size, learning_rate, epsilon, dropout_rate, size_val)

    # save trained model to a file
    with open(f'results/param/{model_file_name}', "wb") as handle:
        pickle.dump(model.neural_networks, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # testing
    accuracy = calculate_accuracy(model, val_dataset)[0]

    # save results to a summary file
    information = {
        "algorithm": "DeepStochLog",
        "seed": seed,
        "nb_epochs": epoch + 1,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epsilon": epsilon,
        "dropout_rate": dropout_rate,
        "size_val": size_val,
        "accuracy": accuracy,
        "model_file": model_file_name
    }
    with open(f'results/summary_param.json', "a") as outfile:
        json.dump(information, outfile)
        outfile.write('\n')

    # print results
    print("############################################")
    print("Accuracy: {}".format(accuracy))
    print("############################################")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        counter = 0
    else:
        if counter >= 2:
            break
        counter += 1