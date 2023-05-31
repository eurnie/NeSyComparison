import sys
import os
import json
import random
import numpy
import time
import torch
from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.heuristics import geometric_mean, ucs
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.evaluate import get_confusion_matrix
from import_data import create_docs_too_much_cites, import_datasets, citeseer_examples, cora_examples

sys.path.append("..")
from data.network_torch import Net_CiteSeer, Net_Cora, Net_PubMed

def calculate_accuracy(dataset, model, data, split):
    _, _, val_counter, test_counter = create_docs_too_much_cites(dataset)
    accuracy = get_confusion_matrix(model, data).accuracy()
    correct_docs = accuracy * len(data)
    print("correct_docs:", correct_docs)
    if split == "val":
        total_docs = len(data) + val_counter
    elif split == "test":
        total_docs = len(data) + test_counter
    print("total docs:", total_docs)
    new_accuracy = correct_docs / total_docs
    return new_accuracy

############################################### PARAMETERS ##############################################
method = "exact"
nb_epochs = 100
batch_size = 2
learning_rate = 0.0001
dropout_rate = 0.2
rely_on_nn = 0.4
#########################################################################################################

for dataset, to_unsupervised in [("Cora", 0)]:
    assert (dataset == "CiteSeer") or (dataset == "Cora") or (dataset == "PubMed")

    for seed in [3, 4, 6, 8, 9]:
        # generate name of file that holds the trained model
        model_file_name = "DeepProbLog_final_{}_{}_{}_{}_{}_{}".format(seed, 
            nb_epochs, batch_size, learning_rate, dropout_rate, rely_on_nn)
        model_file_location = f'results/{method}/{dataset}/final/to_unsupervised_{to_unsupervised}/{model_file_name}'

        if os.path.isfile(model_file_location):
            # setting seeds for reproducibility
            random.seed(seed)
            numpy.random.seed(seed)
            torch.manual_seed(seed)

            # import train, val and test set
            train_set, val_set, test_set = import_datasets(dataset, to_unsupervised, seed)

            if dataset == "CiteSeer":
                network = Net_CiteSeer(dropout_rate)
            elif dataset == "Cora":
                network = Net_Cora(dropout_rate)
            elif dataset == "PubMed":
                network = Net_PubMed(dropout_rate)

            net = Network(network, "document_net", batching=True)
            net.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

            if rely_on_nn is not None:
                model = Model(f'{dataset}_documents_{rely_on_nn}.pl', [net])
            else:
                model = Model(f'{dataset}_documents.pl', [net])
            if method == "exact":
                model.set_engine(ExactEngine(model), cache=False)
            elif method == "geometric_mean":
                model.set_engine(ApproximateEngine(model, 1, geometric_mean, exploration=False))   

            if dataset == "CiteSeer":
                model.add_tensor_source("citeseer", citeseer_examples)
            elif dataset == "Cora":
                model.add_tensor_source("cora", cora_examples)
            loader = DataLoader(train_set, batch_size, False)

            # save trained model to a file
            model.load_state(model_file_location)

            # testing
            start_time = time.time()
            accuracy = calculate_accuracy(dataset, model, test_set, "test")
            testing_time = time.time() - start_time
            
            # save results to a summary file
            information = {
                "algorithm": "DeepProbLog",
                "seed": seed,
                "method": method,
                "nb_epochs": nb_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "dropout_rate": dropout_rate,
                "rely_on_nn": rely_on_nn,
                "accuracy": accuracy,
                "training_time": None,
                "testing_time": testing_time,
                "model_file": model_file_name
            }
            with open(f'results/{method}/{dataset}/summary_final_{to_unsupervised}.json', "a") as outfile:
                json.dump(information, outfile)
                outfile.write('\n')

            # print results
            print("############################################")
            print("Seed: {} \nAccuracy: {} \nTraining time: {} \nTesting time: {}".format(seed, accuracy, 
                None, testing_time))
            print("############################################")