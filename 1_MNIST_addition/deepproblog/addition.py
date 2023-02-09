import time
import torch
import random
import numpy

from import_data import MNIST_train, MNIST_test, addition
from network import Net
from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model

############################################################################################
SEED_PYTHON = 123
SEED_NUMPY = 456
SEED_TORCH = 789
batch_size = 10
nb_epochs = 3
learning_rate = 1e-3
############################################################################################
method = "exact"
N = 1
pretrain = 0
log_iter = 1000
############################################################################################

# setting seeds for reproducibility
random.seed(SEED_PYTHON)
numpy.random.seed(SEED_NUMPY)
torch.manual_seed(SEED_TORCH)

train_set = addition(1, "train", log_iter)
test_set = addition(1, "test", log_iter)

network = Net()
if pretrain is not None and pretrain > 0:
    network.load_state_dict(torch.load("models/pretrained/all_{}.pth".format(pretrain)))
net = Network(network, "mnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

model = Model("addition.pl", [net])
if method == "exact":
    model.set_engine(ExactEngine(model), cache=True)
elif method == "geometric_mean":
    model.set_engine(ApproximateEngine(model, 1, ApproximateEngine.geometric_mean, exploration=False))

model.add_tensor_source("train", MNIST_train)
model.add_tensor_source("test", MNIST_test)

total_training_time = 0
highest_accuracy = 0
highest_accuracy_index = 0

for epoch in range(0, nb_epochs):
    for nb in range(0, len(train_set)):
        loader = DataLoader(train_set[nb], batch_size, False)
        
        # training
        start_time = time.time()
        train = train_model(model, loader, 1, log_iter=10000, profile=0)
        total_training_time += time.time() - start_time

        # testing
        model.eval()
        correct_predictions = 0

        for i, query in enumerate(test_set.to_queries()):
            test_query = query.variable_output()
            answer = model.solve([test_query])[0]
            actual = str(query.output_values()[0])

            if len(answer.result) == 0:
                predicted = None
            else:
                max_ans = max(answer.result, key=lambda x: answer.result[x])
                predicted = str(max_ans.args[query.output_ind[0]])
            if (predicted == actual):
                correct_predictions += 1

        accuracy = correct_predictions / len(test_set)

        if (accuracy > highest_accuracy):
            highest_accuracy = accuracy
            highest_accuracy_index = (epoch * 30000) + log_iter + (nb * log_iter)

        log_file = "results/results_deepproblog_{}_{}_{}_{}_{}_{}.txt".format(SEED_PYTHON, SEED_NUMPY, SEED_TORCH, batch_size, nb_epochs, learning_rate)

        with open(log_file, "a") as f:
            f.write(str((epoch * 30000) + log_iter + (nb * log_iter)))
            f.write(" ")
            f.write(str(total_training_time))
            f.write(" ")
            f.write(str(accuracy))
            f.write(" ")
            f.write("\n")

        print("############################################")
        print("Number of entries: ", (epoch * 30000) + log_iter + (nb * log_iter))
        print("Total training time: ", total_training_time)
        print("Accuracy: ", accuracy)
        print("############################################")

print("The highest accuracy was {} and was reached (the first time) after seeing {} samples.".format(highest_accuracy, highest_accuracy_index))