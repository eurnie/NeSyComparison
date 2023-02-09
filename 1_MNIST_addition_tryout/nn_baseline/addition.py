import torch
import sys
from torch import nn
from torch.utils.data import DataLoader
from network import NeuralNetwork

sys.path.append('../')
from data.import_data import addition_with_only_one_x_value

####################
# hyperparameters
####################

model = NeuralNetwork()
print(model)

batch_size = 1
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

####################
# setup
####################

training_set = addition_with_only_one_x_value(1, "train")
testing_set = addition_with_only_one_x_value(1, "test")

train_dataloader = DataLoader(training_set, batch_size=batch_size)
test_dataloader = DataLoader(testing_set, batch_size=batch_size)

####################
# training and testing
####################

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):

        # compute prediction error
        pred = model(x[0])
        loss = loss_fn(pred, y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x[0])
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 10
for t in range(epochs):
    print("Epoch {}\n-------------------------------".format(t+1))
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)