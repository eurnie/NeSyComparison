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
from data.network_torch import Net, Net_Dropout

############################################### PARAMETERS ##############################################
seed = 0
nb_epochs = 100
batch_size = 8
learning_rate = 0.001
use_dropout = False
#########################################################################################################

DATA_ROOT = Path(__file__).parent.parent.joinpath('data')
data = torch_geometric.datasets.Planetoid(root=str(DATA_ROOT), name="CiteSeer", split="full")
citation_graph = data[0]

# # generate and shuffle dataset
# dataset = [(indices[i], x[i], y[i]) for i in range(len(x))]
# rng = random.Random(seed)
# rng.shuffle(dataset)

# setting seeds for reproducibility
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)

trainDataset = []
valDataset = []

complete_dataset = []
complete_dataset = torch.tensor([])

# import train and val set
for i in range(round(len(citation_graph.x))):
    if citation_graph.train_mask[i]:
        trainDataset.append((i, citation_graph.y[i]))
    elif citation_graph.val_mask[i]:
        valDataset.append((i, citation_graph.y[i]))
    complete_dataset = torch.cat((complete_dataset, citation_graph.x[i].unsqueeze(0)), dim=0)

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
if use_dropout:
    m = Net_Dropout()
else:
    m = Net()
nnMapping = {'document_label_neural': m}
optimizers = {'document_label_neural': torch.optim.Adam(m.parameters(), lr=learning_rate)}
NeurASPobj = NeurASP(dprogram, nnMapping, optimizers)

# generate name of file that holds the trained model
model_file_name = "NeurASP_param_{}_{}_{}_{}_{}".format(seed, nb_epochs, batch_size, 
    learning_rate, use_dropout)

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
    "use_dropout": use_dropout,
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