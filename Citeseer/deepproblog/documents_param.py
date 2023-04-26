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
from import_data import import_datasets, citeseer_examples, citeseer_cites

sys.path.append("..")
from data.network_torch import Net, Net_Dropout

############################################### PARAMETERS ##############################################
seed = 0
method = "exact"
nb_epochs = 100
batch_size = 32
learning_rate = 0.001
use_dropout = False
#########################################################################################################

for method in ["geometric_mean", "exact"]:
    for batch_size in [2, 4, 8, 16, 32, 64]:
        # setting seeds for reproducibility
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)

        # import train and val set
        train_set, val_set, _ = import_datasets(seed)

        if use_dropout:
            network = Net_Dropout()
        else:
            network = Net()

        net = Network(network, "citeseer_net", batching=True)
        net.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

        model = Model("documents.pl", [net])
        if method == "exact":
            model.set_engine(ExactEngine(model), cache=False)
        elif method == "geometric_mean":
            model.set_engine(ApproximateEngine(model, 1, geometric_mean, exploration=False))   

        model.add_tensor_source("citeseer", citeseer_examples)
        model.add_tensor_source("linked", citeseer_cites)
        loader = DataLoader(train_set, batch_size, False)

        best_accuracy = 0

        # training
        for epoch in range(nb_epochs):
            train_model(model, loader, 1, log_iter=100, profile=0)

            # generate name of file that holds the trained model
            model_file_name = "DeepProbLog_param_{}_{}_{}_{}_{}_{}".format(seed, method, epoch + 1, batch_size, 
                learning_rate, use_dropout)

            # save trained model to a file
            model.save_state(f'results/{method}/param/{model_file_name}')

            # testing
            accuracy = get_confusion_matrix(model, val_set, verbose=0).accuracy()

            # save results to a summary file
            information = {
                "algorithm": "DeepProbLog",
                "seed": seed,
                "method": method,
                "nb_epochs": epoch + 1,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "use_dropout": use_dropout,
                "accuracy": accuracy,
                "model_file": model_file_name
            }
            with open(f'results/{method}/summary_param.json', "a") as outfile:
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