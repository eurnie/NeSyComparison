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
from program import CiteSeer_dprogram, Cora_dprogram, PubMed_dprogram
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
            if counter >= 2:
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
batch_size = 2
learning_rate = 0.001
dropout_rate = 0
#########################################################################################################

for seed in range(0, 10):
    # setting seeds for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    if dataset == "CiteSeer":
        program = CiteSeer_dprogram
    elif dataset == "Cora":
        program = Cora_dprogram
    elif dataset == "PubMed":
        program = PubMed_dprogram

    # import dataset
    DATA_ROOT = Path(__file__).parent.parent.joinpath('data')
    data = torch_geometric.datasets.Planetoid(root=str(DATA_ROOT), name="CiteSeer", split="full")
    citation_graph = data[0]

    cites_a = citation_graph.edge_index[0]
    cites_b = citation_graph.edge_index[1]

    trainDataset = []
    valDataset = []
    testDataset = []
    ind_to_labels_train = []
    ind_to_labels_val = []
    ind_to_labels_test = []
    ind_to_features = torch.tensor([])

    # import train and val set
    for i in range(round(len(citation_graph.x))):
        if citation_graph.train_mask[i]:
            trainDataset.append((i, citation_graph.x[i].unsqueeze(0), citation_graph.y[i]))
            ind_to_labels_train.append(citation_graph.y[i])
            ind_to_labels_val.append('no_label')
            ind_to_labels_test.append('no_label')
        elif citation_graph.val_mask[i]:
            valDataset.append((i, citation_graph.x[i].unsqueeze(0), citation_graph.y[i]))
            ind_to_labels_train.append('no_label')
            ind_to_labels_val.append(citation_graph.y[i])
            ind_to_labels_test.append('no_label')
        elif citation_graph.test_mask[i]:
            testDataset.append((i, citation_graph.x[i].unsqueeze(0), citation_graph.y[i]))
            ind_to_labels_train.append('no_label')
            ind_to_labels_val.append('no_label')
            ind_to_labels_test.append(citation_graph.y[i])

        ind_to_features = torch.cat((ind_to_features, citation_graph.x[i].unsqueeze(0)), dim=0)
        
    # move train examples to the test set according to the given ratio
    if move_to_test_set_ratio > 0:
        split_index = round(move_to_test_set_ratio * len(trainDataset))
        
        for elem in trainDataset[:split_index]:
            testDataset.append(elem)
        trainDataset = trainDataset[split_index:]

        for elem in ind_to_labels_train[:split_index]:
            ind_to_labels_test.append(elem)
        ind_to_labels_train = ind_to_labels_train[split_index:]
        
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
                    dataList_train.append({'doc_1': doc_1, 'doc_2': ind_to_features[cites_b[i]].unsqueeze(0)})
                    obsList_train.append(f':- not document_label(doc_1, doc_2, {label_1}, {ind_to_labels_train[cites_b[i]]}, 1).')
                    is_cited = True

        if not is_cited:
            dataList_train.append({'doc_1': doc_1, 'doc_2': dummy_doc})
            obsList_train.append(f':- not document_label(doc_1, empty, {label_1}, no_label, 0).')

    dataList_val = []
    obsList_val = []
    for index_1, doc_1, label_1 in valDataset:
        dataList_val.append({'doc_1': doc_1, 'doc_2': dummy_doc})
        obsList_val.append(f':- not document_label(doc_1, empty, {label_1}, no_label, 0).')

    assert len(valDataset) == len(dataList_val)

    dataList_test = []
    obsList_test = []
    for index_1, doc_1, label_1 in testDataset:
        dataList_test.append({'doc_1': doc_1, 'doc_2': dummy_doc})
        obsList_test.append(f':- not document_label(doc_1, empty, {label_1}, no_label, 0).')
 
    assert len(testDataset) == len(dataList_test)

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
    model_file_name = "NeurASP_final_{}_{}_{}_{}_{}_{}".format(seed, nb_epochs, batch_size, 
        learning_rate, dropout_rate, move_to_test_set_ratio)

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
    with open(f'results/{dataset}/summary_final_{move_to_test_set_ratio}.json', "a") as outfile:
        json.dump(information, outfile)
        outfile.write('\n')

    # print results
    print("############################################")
    print("Seed: {} \nAccuracy: {} \nTraining time: {} \nTesting time: {}".format(seed, accuracy, 
        training_time, testing_time))
    print("############################################")