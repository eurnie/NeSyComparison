import os
import sys
import json
import random
import numpy
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

################################################# DATASET ###############################################
dataset = "CiteSeer"
#########################################################################################################

############################################### PARAMETERS ##############################################
seed = 0
nb_epochs = 100
method = 'exact'
#########################################################################################################

assert (dataset == "CiteSeer") or (dataset == "Cora") or (dataset == "PubMed")

for dropout_rate in [0, 0.2]:
    for rely_on_nn in [None, 0.4]:
        for learning_rate in [0.001, 0.0001]:
            for batch_size in [2, 8, 32, 128]:
                # generate name of file that holds the trained model
                model_file_name = "DeepProbLog_param_{}_{}_{}_{}_{}_{}".format(seed, 
                    nb_epochs, batch_size, learning_rate, dropout_rate, rely_on_nn)
                model_file_location = f'results/{method}/{dataset}/param/{model_file_name}'

                if not os.path.isfile(model_file_location):
                    # setting seeds for reproducibility
                    random.seed(seed)
                    numpy.random.seed(seed)
                    torch.manual_seed(seed)

                    # import train and val set
                    train_set, val_set, _ = import_datasets(dataset, 0, seed)

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

                    if not os.path.exists("best_model"):
                        os.mkdir("best_model")

                    # training (with early stopping)
                    total_training_time = 0
                    best_accuracy = -1
                    counter = 0
                    nb_epochs_done = 0
                    for epoch in range(nb_epochs):
                        train = train_model(model, loader, 1, log_iter=100, profile=0)
                        val_accuracy = calculate_accuracy(dataset, train.model, val_set, "val")
                        print("Val accuracy after epoch", epoch, ":", val_accuracy)
                        if val_accuracy > best_accuracy:
                            best_accuracy = val_accuracy
                            train.model.save_state("best_model/state", complete=True)
                            counter = 0
                            nb_epochs_done = epoch + 1
                        else:
                            if counter >= 2:
                                break
                            counter += 1

                    # early stopping: load best model and delete file
                    model.load_state("best_model/state")
                    os.remove("best_model/state")
                    os.rmdir("best_model")

                    # save trained model to a file
                    model.save_state(model_file_location)

                    # save results to a summary file
                    information = {
                        "algorithm": "DeepProbLog",
                        "seed": seed,
                        "method": method,
                        "nb_epochs": nb_epochs_done,
                        "batch_size": batch_size,
                        "rely_on_nn": rely_on_nn,
                        "learning_rate": learning_rate,
                        "dropout_rate": dropout_rate,
                        "accuracy": best_accuracy,
                        "model_file": model_file_name
                    }
                    with open(f'results/{method}/{dataset}/summary_param.json', "a") as outfile:
                        json.dump(information, outfile)
                        outfile.write('\n')

                    # print results
                    print("############################################")
                    print("Accuracy: {}".format(best_accuracy))
                    print("############################################")