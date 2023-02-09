import sys
import torch
from torch.utils.data import DataLoader
from network import Net
from neurasp.neurasp import NeurASP

# retrieve the training and testing data

sys.path.append('../')
from data.import_data import addition

training_set = addition(1, "train")
testing_set = addition(1, "test")

train_dataloader = DataLoader(training_set, batch_size=1)
test_dataloader = DataLoader(testing_set, batch_size=1)

dataList = []
obsList = []
for i1, i2, l in training_set:
    dataList.append({'i1': i1[0].unsqueeze(0), 'i2': i2[0].unsqueeze(0)})
    obsList.append(':- not addition(i1, i2, {}).'.format(l))

dataList_test = []
obsList_test = []
for i1, i2, l in testing_set:
    dataList_test.append({'i1': i1[0].unsqueeze(0), 'i2': i2[0].unsqueeze(0)})
    obsList_test.append(':- not addition(i1, i2, {}).'.format(l))

# NeurASP program

dprogram = '''
img(i1). img(i2).
addition(A,B,N) :- digit(0,A,N1), digit(0,B,N2), N=N1+N2.
nn(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).
'''

# define nnMapping and optimizers, initialze NeurASP object

m = Net()
nnMapping = {'digit': m}
optimizers = {'digit': torch.optim.Adam(m.parameters(), lr=0.001)}
NeurASPobj = NeurASP(dprogram, nnMapping, optimizers)

# training and testing

epochs = 10
for t in range(epochs):
    print("Epoch {}\n-------------------------------".format(t+1))
    NeurASPobj.learn(dataList=dataList, obsList=obsList, epoch=1, smPickle=None, bar=True)

    acc = NeurASPobj.testInferenceResults(dataList_test, obsList_test)
    print("ACCURACY: ", acc)
