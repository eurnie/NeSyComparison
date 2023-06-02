import os
import json
import sys
import time
import random
import numpy
import torch
import pickle
import torch_geometric
from pathlib import Path
from program_citeseer import CiteSeer_dprogram
from program_cora import Cora_dprogram
from program_pubmed import PubMed_dprogram
from neurasp.neurasp import NeurASP

sys.path.append("..")
from data.network_torch import Net_CiteSeer, Net_Cora, Net_PubMed

############################################### PARAMETERS ##############################################
nb_epochs = 100
method = "exact"
batch_size = 128
learning_rate = 0.0001
opt = False
dropout_rate = 0
#########################################################################################################

for dataset, to_unsupervised in [("CiteSeer", 0.1)]:
    assert (dataset == "CiteSeer") or (dataset == "Cora") or (dataset == "PubMed")

    if dataset == "CiteSeer":
        program = CiteSeer_dprogram
    elif dataset == "Cora":
        program = Cora_dprogram
    elif dataset == "PubMed":
        program = PubMed_dprogram

    # import dataset
    DATA_ROOT = Path(__file__).parent.parent.joinpath('data')
    data = torch_geometric.datasets.Planetoid(root=str(DATA_ROOT), name=dataset, split="full")
    citation_graph = data[0]

    cites_a = citation_graph.edge_index[0]
    cites_b = citation_graph.edge_index[1]

    for seed in [5]:
        # generate name of file that holds the trained model
        model_file_name = "NeurASP_final_{}_{}_{}_{}_{}_{}".format(seed, 
            nb_epochs, batch_size, learning_rate, dropout_rate, opt)
        model_file_location = f'results/{method}/{dataset}/final/to_unsupervised_{to_unsupervised}/{model_file_name}'

        if os.path.isfile(model_file_location):
            # setting seeds for reproducibility
            random.seed(seed)
            numpy.random.seed(seed)
            torch.manual_seed(seed)

            trainDataset = []
            valDataset = []
            testDataset = []
            ind_to_features = torch.tensor([])

            # import train and val set
            for i in range(round(len(citation_graph.x))):
                if citation_graph.train_mask[i]:
                    trainDataset.append((i, citation_graph.x[i].unsqueeze(0), citation_graph.y[i]))
                elif citation_graph.val_mask[i]:
                    valDataset.append((i, citation_graph.x[i].unsqueeze(0), citation_graph.y[i]))
                elif citation_graph.test_mask[i]:
                    testDataset.append((i, citation_graph.x[i].unsqueeze(0), citation_graph.y[i]))

                ind_to_features = torch.cat((ind_to_features, citation_graph.x[i].unsqueeze(0)), dim=0)
                
            # move train examples to the test set according to the given ratio
            if to_unsupervised > 0:
                split_index = round(to_unsupervised * len(trainDataset))
                trainDataset = trainDataset[split_index:]
                
            print("The training set contains", len(trainDataset), "instances.")
            print("The validation set contains", len(valDataset), "instances.")
            print("The testing set contains", len(testDataset), "instances.")

            dummy_doc = torch.zeros(1, len(citation_graph.x[0]))

            dataList_train = []
            obsList_train = []
            for index_1, doc_1, label_1 in trainDataset:
                is_cited = False
                for i in range(0, len(cites_a)):
                    if (cites_a[i] == index_1):
                            index_2 = cites_b[i].item()
                            dataList_train.append({'doc_1': doc_1, 'doc_2': ind_to_features[index_2].unsqueeze(0)})
                            obs_string = ''
                            obs_string += ':- not label({},doc_1,{}).'.format(int(index_1), int(label_1))
                            obs_string += '\nind_to_doc({},doc_1).'.format(int(index_1))
                            obs_string += '\nind_to_doc({},doc_2).'.format(int(index_2))
                            obsList_train.append(obs_string)
                            is_cited = True

                if not is_cited:
                    dataList_train.append({'doc_1': doc_1, 'doc_2': dummy_doc})
                    obs_string = ''
                    obs_string += ':- not label({},doc_1,{}).'.format(int(index_1), int(label_1))
                    obs_string += '\nind_to_doc({},doc_1).'.format(int(index_1))
                    obsList_train.append(obs_string)

            dataList_val = []
            obsList_val = []
            extra_rules_val = []
            for index_1, doc_1, label_1 in valDataset:
                dataList_val.append({'doc_1': doc_1, 'doc_2': dummy_doc})
                obs_string = ''
                obs_string += ':- not label({},doc_1,{}).'.format(int(index_1), int(label_1))
                obs_string += '\nind_to_doc({},doc_1).'.format(int(index_1))
                obsList_val.append(obs_string)
                rules_string = ''
                rules_string += '\nind_to_doc({},doc_1).'.format(int(index_1))
                extra_rules_val.append(rules_string)
            assert len(valDataset) == len(dataList_val)

            dataList_test = []
            obsList_test = []
            extra_rules_test = []
            for index_1, doc_1, label_1 in testDataset:
                dataList_test.append({'doc_1': doc_1, 'doc_2': dummy_doc})
                obs_string = ''
                obs_string += ':- not label({},doc_1,{}).'.format(int(index_1), int(label_1))
                obs_string += '\nind_to_doc({},doc_1).'.format(int(index_1))
                obsList_test.append(obs_string)
                rules_string = ''
                rules_string += '\nind_to_doc({},doc_1).'.format(int(index_1))
                extra_rules_test.append(rules_string)
            assert len(testDataset) == len(dataList_test)

            # define nnMapping and optimizers, initialze NeurASP object
            if dataset == "CiteSeer":
                m = Net_CiteSeer(dropout_rate)
            elif dataset == "Cora":
                m = Net_Cora(dropout_rate)
            elif dataset == "PubMed":
                m = Net_PubMed(dropout_rate)
            nnMapping = {'label_neural': m}
            optimizers = {'label_neural': torch.optim.Adam(m.parameters(), lr=learning_rate)}
            NeurASPobj = NeurASP(program, nnMapping, optimizers)

            with open(model_file_location, "rb") as handle:
                NeurASPobj = pickle.load(handle)

            os.remove("best_model.pickle")

            # save trained model to a file
            with open(model_file_location, "wb") as handle:
                pickle.dump(NeurASPobj, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # testing
            start_time = time.time()
            accuracy = NeurASPobj.testInferenceResults(dataList_test, obsList_test, extra_rules=extra_rules_test) / 100
            testing_time = time.time() - start_time

            # save results to a summary file
            information = {
                "algorithm": "NeurASP",
                "seed": seed,
                "nb_epochs_done": nb_epochs_done,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "dropout_rate": dropout_rate,
                "accuracy": accuracy,
                "training_time": total_training_time,
                "testing_time": testing_time,
                "model_file": model_file_name
            }
            with open(f'results/{method}/{dataset}/summary_final_{to_unsupervised}.json', "a") as outfile:
                json.dump(information, outfile)
                outfile.write('\n')

            # print results
            print("############################################")
            print("Seed: {} \nAccuracy: {} \nTraining time: {} \nTesting time: {}".format(seed, accuracy, 
                total_training_time, testing_time))
            print("############################################")