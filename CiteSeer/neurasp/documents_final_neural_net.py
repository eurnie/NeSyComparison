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
from program import CiteSeer_dprogram_only_neural_network, Cora_dprogram_only_neural_network, PubMed_dprogram_only_neural_network
from neurasp.neurasp import NeurASP

sys.path.append("..")
from data.network_torch import Net_CiteSeer, Net_Cora, Net_PubMed

def train_and_test(model_file_name, dataList_train, obsList_train, dataList_val, obsList_val, 
    dataList_test, obsList_test, nb_epochs, batch_size):
    
    # training (with early stopping)
    total_training_time = 0
    best_accuracy = -1
    counter = 0
    nb_epochs_done = 0

    for epoch in range(nb_epochs):
        start_time = time.time()
        NeurASPobj.learn(dataList=dataList_train, obsList=obsList_train, epoch=1, smPickle=None, 
            bar=True, batchSize=batch_size)
        total_training_time += time.time() - start_time
        val_accuracy = NeurASPobj.testInferenceResults(dataList_val, obsList_val) / 100
        print("Val accuracy after epoch", epoch, ":", val_accuracy)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            with open("best_model.pickle", "wb") as handle:
                pickle.dump(NeurASPobj, handle, protocol=pickle.HIGHEST_PROTOCOL)
            counter = 0
            nb_epochs_done = epoch + 1
        else:
            if counter >= 1:
                break
            counter += 1
    with open("best_model.pickle", "rb") as handle:
        BestNeurASPobj = pickle.load(handle)

    os.remove("best_model.pickle")

    # save trained model to a file
    with open(f'results/{dataset}/final/{model_file_name}', "wb") as handle:
        pickle.dump(NeurASPobj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # testing
    start_time = time.time()
    accuracy = BestNeurASPobj.testInferenceResults(dataList_test, obsList_test) / 100
    testing_time = time.time() - start_time

    return nb_epochs_done, accuracy, total_training_time, testing_time

################################################# DATASET ###############################################
dataset = "CiteSeer"
move_to_test_set_ratio = 0
#########################################################################################################

############################################### PARAMETERS ##############################################
nb_epochs = 100
batch_size = 64
learning_rate = 0.001
dropout_rate = 0
#########################################################################################################

for seed in range(0, 10):
    # setting seeds for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    if dataset == "CiteSeer":
        program = CiteSeer_dprogram_only_neural_network
    elif dataset == "Cora":
        program = Cora_dprogram_only_neural_network
    elif dataset == "PubMed":
        program = PubMed_dprogram_only_neural_network

    # import dataset
    DATA_ROOT = Path(__file__).parent.parent.joinpath('data')
    data = torch_geometric.datasets.Planetoid(root=str(DATA_ROOT), name="CiteSeer", split="full")
    citation_graph = data[0]

    trainDataset = []
    valDataset = []
    testDataset = []
    ind_to_features = torch.tensor([])

    # import train, val and test set
    for i in range(round(len(citation_graph.x))):
        if citation_graph.train_mask[i]:
            trainDataset.append((i, citation_graph.y[i]))
        elif citation_graph.val_mask[i]:
            valDataset.append((i, citation_graph.y[i]))
        elif citation_graph.test_mask[i]:
            testDataset.append((i, citation_graph.y[i]))

        ind_to_features = torch.cat((ind_to_features, citation_graph.x[i].unsqueeze(0)), dim=0)

    # move train examples to the test set according to the given ratio
    if move_to_test_set_ratio > 0:
        split_index = round(move_to_test_set_ratio * len(trainDataset))
        for elem in trainDataset[:split_index]:
            testDataset.append(elem)
        trainDataset = trainDataset[split_index:]

    print("The training set contains", len(trainDataset), "instances.")
    print("The validation set contains", len(valDataset), "instances.")
    print("The testing set contains", len(testDataset), "instances.")

    # dataList_dictionary = {}
    # for i in range(len(ind_to_features)):
    #     dataList_dictionary[f'document_{i}'] = ind_to_features[i].unsqueeze(0)

    # cites_a = citation_graph.edge_index[0]
    # cites_b = citation_graph.edge_index[1]
    # for i in range(0, len(cites_a)):
    #         program += f'cite({cites_a[i]},{cites_b[i]}).\n'

    # for i in range(0, len(ind_to_features)):
    #     program += f'doc(document_{i}).\n'

    # with open("generated_program.txt", "a") as f:
    #     f.write(program)

    dataList_train = []
    obsList_train = []
    for index, label in trainDataset:
        dataList_train.append({'document': ind_to_features[index].unsqueeze(0)})
        obsList_train.append(f':- not document_label(document, {label}).')
        # dataList_train.append(dataList_dictionary)
        # obsList_train.append(f':- not document_label(document_{index}, {label}).')

    dataList_val = []
    obsList_val = []
    for index, label in valDataset:
        dataList_val.append({'document': ind_to_features[index].unsqueeze(0)})
        obsList_val.append(f':- not document_label(document, {label}).')
        # dataList_val.append(dataList_dictionary)
        # obsList_val.append(f':- not document_label(document_{index}, {label}).')

    dataList_test = []
    obsList_test = []
    for index, label in testDataset:
        dataList_test.append({'document': ind_to_features[index].unsqueeze(0)})
        obsList_test.append(f':- not document_label(document, {label}).')
        # dataList_test.append(dataList_dictionary)
        # obsList_test.append(f':- not document_label(document_{index}, {label}).')

    # define nnMapping and optimizers, initialze NeurASP object
    if dataset == "CiteSeer":
        m = Net_CiteSeer(dropout_rate)
    elif dataset == "Cora":
        m = Net_Cora(dropout_rate)
    elif dataset == "PubMed":
        m = Net_PubMed(dropout_rate)
    nnMapping = {'document_label_neural': m}
    optimizers = {'document_label_neural': torch.optim.Adam(m.parameters(), lr=learning_rate)}
    NeurASPobj = NeurASP(program, nnMapping, optimizers)

    # generate name of file that holds the trained model
    model_file_name = "NeurASP_final_{}_{}_{}_{}_{}".format(seed, nb_epochs, batch_size, 
        learning_rate, dropout_rate)

    # train and test
    nb_epochs_done, accuracy, training_time, testing_time = train_and_test(model_file_name, dataList_train, obsList_train,
        dataList_val, obsList_val, dataList_test, obsList_test, nb_epochs, batch_size)

    # save results to a summary file
    information = {
        "algorithm": "NeurASP",
        "seed": seed,
        "nb_epochs_done": nb_epochs_done,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dropout_rate": dropout_rate,
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