import json
import sys
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

################################################# DATASET ###############################################
dataset = "CiteSeer"
move_to_test_set_ratio = 0
#########################################################################################################

############################################### PARAMETERS ##############################################
seed = 0
nb_epochs = 100
batch_size = 64
learning_rate = 0.001
dropout_rate = 0
#########################################################################################################

assert (dataset == "CiteSeer") or (dataset == "Cora") or (dataset == "PubMed")

for batch_size in [2, 4, 8, 16, 32, 64]:
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

    DATA_ROOT = Path(__file__).parent.parent.joinpath('data')
    data = torch_geometric.datasets.Planetoid(root=str(DATA_ROOT), name=dataset, split="full")
    citation_graph = data[0]

    cites_a = citation_graph.edge_index[0]
    cites_b = citation_graph.edge_index[1]

    trainDataset = []
    valDataset = []
    ind_to_labels_train = []
    ind_to_labels_val = []
    ind_to_features = torch.tensor([])

    # import train and val set
    for i in range(round(len(citation_graph.x))):
        if citation_graph.train_mask[i]:
            trainDataset.append((i, citation_graph.x[i].unsqueeze(0), citation_graph.y[i]))
            ind_to_labels_train.append(citation_graph.y[i])
            ind_to_labels_val.append('no_label')
        elif citation_graph.val_mask[i]:
            valDataset.append((i, citation_graph.x[i].unsqueeze(0), citation_graph.y[i]))
            ind_to_labels_train.append('no_label')
            ind_to_labels_val.append(citation_graph.y[i])
        elif citation_graph.test_mask[i]:
            ind_to_labels_train.append('no_label')
            ind_to_labels_val.append('no_label')

        ind_to_features = torch.cat((ind_to_features, citation_graph.x[i].unsqueeze(0)), dim=0)
        
    # shuffle dataset
    rng = random.Random(seed)
    rng.shuffle(trainDataset)
    rng = random.Random(seed)
    rng.shuffle(valDataset)

    print("The training set contains", len(trainDataset), "instances.")
    print("The validation set contains", len(valDataset), "instances.")

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

    best_accuracy = 0

    # train and test
    for epoch in range(nb_epochs):
        # training
        NeurASPobj.learn(dataList=dataList_train, obsList=obsList_train, epoch=1, smPickle=None, 
            bar=True, batchSize=batch_size)
        
        # generate name of file that holds the trained model
        model_file_name = "NeurASP_param_{}_{}_{}_{}_{}".format(seed, 
            epoch + 1, batch_size, learning_rate, dropout_rate)

        # save trained model to a file
        with open(f'results/{dataset}/param/{model_file_name}', "wb") as handle:
            pickle.dump(NeurASPobj, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # testing
        accuracy = NeurASPobj.testInferenceResults(dataList_val, obsList_val) / 100

        # save results to a summary file
        information = {
            "algorithm": "NeurASP",
            "seed": seed,
            "nb_epochs": epoch + 1,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dropout_rate": dropout_rate,
            "accuracy": accuracy,
            "model_file": model_file_name
        }
        with open(f'results/{dataset}/summary_param.json', "a") as outfile:
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