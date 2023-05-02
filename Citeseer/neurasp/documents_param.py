import json
import sys
import random
import numpy
import torch
import pickle
import torch_geometric
from pathlib import Path
from program import dprogram
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
batch_size = 512
learning_rate = 0.001
dropout_rate = 0
#########################################################################################################

assert (dataset == "CiteSeer") or (dataset == "Cora") or (dataset == "PubMed")

DATA_ROOT = Path(__file__).parent.parent.joinpath('data')
data = torch_geometric.datasets.Planetoid(root=str(DATA_ROOT), name=dataset, split="full")
citation_graph = data[0]

# setting seeds for reproducibility
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)

trainDataset = []
valDataset = []
testDataset = []

complete_dataset_train = torch.tensor([])
complete_dataset_val = torch.tensor([])
complete_dataset_test = torch.tensor([])

# import train, val and test set
for i in range(round(len(citation_graph.x))):
    if citation_graph.train_mask[i]:
        trainDataset.append((i, citation_graph.y[i]))
    elif citation_graph.val_mask[i]:
        valDataset.append((i, citation_graph.y[i]))
    elif citation_graph.test_mask[i]:
        testDataset.append((i, citation_graph.y[i]))

        
    complete_dataset = torch.cat((complete_dataset, citation_graph.x[i].unsqueeze(0)), dim=0)

# move train examples to the test set according to the given ratio
if dataset_name == "train":
    if move_to_test_set_ratio > 0:
        split_index = round(move_to_test_set_ratio * len(dataset))
        train_set = dataset[split_index:]
        for elem in dataset[:split_index]:
            test_set_to_add.append(elem)
    else:
        train_set = dataset
elif dataset_name == "val":
    val_set = dataset
elif dataset_name == "test":
    test_set = dataset
    for elem in test_set_to_add:
        test_set.append(elem)

# shuffle dataset
rng = random.Random(seed)
rng.shuffle(trainDataset)
rng = random.Random(seed)
rng.shuffle(valDataset)

dataList_train = []
obsList_train = []
for index, label in trainDataset:
    dataList_train.append({'ind': index, 'citeseer': complete_dataset})
    obsList_train.append(f':- not document_label(ind, {label}).')

dataList_val = []
obsList_val = []
for index, label in valDataset:
    dataList_val.append({'ind': index, 'citeseer': complete_dataset})
    obsList_val.append(f':- not document_label(ind, {label}).')

# define nnMapping and optimizers, initialze NeurASP object
if dataset == "CiteSeer":
    m = Net_CiteSeer(dropout_rate)
elif dataset == "Cora":
    m = Net_Cora(dropout_rate)
elif dataset == "PubMed":
    m = Net_PubMed(dropout_rate)
nnMapping = {'document_label_neural': m}
optimizers = {'document_label_neural': torch.optim.Adam(m.parameters(), lr=learning_rate)}
NeurASPobj = NeurASP(dprogram, nnMapping, optimizers)

# generate name of file that holds the trained model
model_file_name = "NeurASP_param_{}_{}_{}_{}_{}".format(seed, nb_epochs, batch_size, 
    learning_rate, dropout_rate)

# train and test
# training
NeurASPobj.learn(dataList=dataList_train, obsList=obsList_train, epoch=nb_epochs, smPickle=None, 
    bar=True, batchSize=batch_size)

# save trained model to a file
with open("results/param/{}".format(model_file_name), "wb") as handle:
    pickle.dump(NeurASPobj, handle, protocol=pickle.HIGHEST_PROTOCOL)

# testing
accuracy = NeurASPobj.testInferenceResults(dataList_val, obsList_val) / 100

# save results to a summary file
information = {
    "algorithm": "NeurASP",
    "seed": seed,
    "nb_epochs": nb_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "dropout_rate": dropout_rate,
    "accuracy": accuracy,
    "model_file": model_file_name
}
with open("results/summary_param.json", "a") as outfile:
    json.dump(information, outfile)
    outfile.write('\n')

# print results
print("############################################")
print("Accuracy: {}".format(accuracy))
print("############################################")