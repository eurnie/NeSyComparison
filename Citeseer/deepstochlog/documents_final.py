import random
import numpy
import json
import os
import time
import pickle
import torch
import sys
from pathlib import Path
from torch.optim import Adam
from deepstochlog.network import Network, NetworkStore
from import_data import get_datasets
from deepstochlog.dataloader import DataLoader
from deepstochlog.model import DeepStochLogModel
from deepstochlog.trainer import DeepStochLogTrainer, print_logger
from deepstochlog.term import Term
from citeseer_utils import create_model_accuracy_calculator, calculate_accuracy

sys.path.append("..")
from data.network_torch import Net_CiteSeer, Net_Cora, Net_PubMed

def train_and_test(dataset, model_file_name, train_set, val_set, test_set, nb_epochs, batch_size, 
                   learning_rate, epsilon, dropout_rate):
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
        prolog_facts= citations,
        verbose=False
    )
    proving_time = time.time() - proving_start

    optimizer = Adam(model.get_all_net_parameters(), lr=learning_rate)
    optimizer.zero_grad()
    logger.print("\nProving the program took {:.2f} seconds".format(proving_time))

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    dummy_dataloader = DataLoader([], batch_size=1, shuffle=False)

    # create test functions
    calculate_model_accuracy = create_model_accuracy_calculator(model, dummy_dataloader,  time.time())

    # training (with early stopping)
    trainer = DeepStochLogTrainer(log_freq=10, accuracy_tester=calculate_model_accuracy)
    total_training_time = 0
    best_accuracy = 0
    counter = 0
    for epoch in range(nb_epochs):
        total_training_time += trainer.train(model, optimizer, train_dataloader, 1, epsilon)
        print(f'Training for epoch {epoch + 1} done.')
        val_accuracy = calculate_accuracy(model, val_set)[0]
        print("Val accuracy after epoch", epoch + 1, ":", val_accuracy)
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

    os.remove("best_model.pickle")

    # save trained model to a file
    with open(f'results/{dataset}/final/{model_file_name}', "wb") as handle:
        pickle.dump(model.neural_networks, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # testing
    start_time = time.time()
    accuracy = calculate_accuracy(model, test_set)[0]
    testing_time = time.time() - start_time

    return accuracy, total_training_time, testing_time

################################################# DATASET ###############################################
dataset = "CiteSeer"
move_to_test_set_ratio = 0
#########################################################################################################

############################################### PARAMETERS ##############################################
nb_epochs = 100
batch_size = 32
learning_rate = 0.001
epsilon = 0.00000001
dropout_rate = 0
size_val = 0.1
#########################################################################################################

logger = print_logger

for seed in range(0, 10):
    # setting seeds for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    train_dataset, val_dataset, test_dataset, queries_for_model, citations = get_datasets(dataset, move_to_test_set_ratio, seed)

    # generate name of file that holds the trained model
    model_file_name = "DeepStochLog_final_{}_{}_{}_{}_{}_{}".format(seed, nb_epochs, 
        batch_size, learning_rate, epsilon, dropout_rate, size_val)

    # train and test
    nb_epochs_done, accuracy, training_time, testing_time = train_and_test(dataset, model_file_name, 
        train_dataset, val_dataset, test_dataset, nb_epochs, batch_size, learning_rate, epsilon, dropout_rate)

    # save results to a summary file
    information = {
        "algorithm": "DeepStochLog",
        "seed": seed,
        "nb_epochs": nb_epochs_done,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epsilon": epsilon,
        "dropout_rate": dropout_rate,
        "size_val": size_val,
        "accuracy": accuracy,
        "training_time": training_time,
        "testing_time": testing_time,
        "model_file": model_file_name
    }
    with open(f'results/{dataset}/summary_final.json', "a") as outfile:
        json.dump(information, outfile)
        outfile.write('\n')

    # print results
    print("############################################")
    print("Seed: {} \nAccuracy: {} \nTraining time: {} \nTesting time: {}".format(seed, accuracy, 
        training_time, testing_time))
    print("############################################")