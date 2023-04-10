import cv2
import random
import time
import sys
import os
import json
import torch
import pickle
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from semantic_loss_pytorch import SemanticLoss

sys.path.append("..")
from data.generate_dataset import generate_dataset
from data.display_map import display_map, display_map_and_path
from data.network_torch import Net_NN, Net_NN_Dropout

def train(dataloader, model, sl, loss_fn, optimizer):
    model.train()

    for (x, y) in dataloader:
        # compute prediction error
        pred = model(x)
        y = y.view(-1, 144)
        loss = loss_fn(pred, y) + sl(pred)
        print(loss.item())

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def valid_coordinates(nb_rows, nb_columns, x, y):
    if x >= 0 and x < nb_columns and y >= 0 and y < nb_rows:
        return True
    else:
        return False
    
def construct_path(path_matrix):
    possible_moves = [(1,0), (-1,0), (0,1), (0,-1), (-1,-1), (1,-1), (-1,1), (1,1)]
    previous_i = 0
    previous_j = 0
    constructed_path = [(previous_i, previous_j)]

    while previous_i != len(path_matrix) - 1 or previous_j != len(path_matrix[0]) - 1:
        found = False
        for move_i, move_j in possible_moves:
            new_i = previous_i + move_i
            new_j = previous_j + move_j

            if valid_coordinates(len(path_matrix), len(path_matrix[0]), new_i, new_j):
                if path_matrix[new_i][new_j] == 1 and (new_i, new_j) not in constructed_path:
                    constructed_path.append((new_i, new_j))
                    found = True
                    break
        
        if found:
            previous_i = new_i
            previous_j = new_j
        else:
            return None

    return constructed_path

# returns two boolean values:
# - first one indicates if the given path is a valid path (starts in the left upper corner and ends in the 
# right lower corner and only does legal moves)
# - second one indicates if the given path has the same cost as the shortest path that was given (this means
# the given path is the/a shortest path)
def same_path_cost(cost_matrix, predicted, real):
    path_predicted = construct_path(predicted)
    path_real = construct_path(real)

    if path_predicted is not None:
        cost_predicted = 0
        cost_real = 0
        for (i, j) in path_predicted:
            cost_predicted += cost_matrix[i][j]
        for (i, j) in path_real:
            cost_real += cost_matrix[i][j]
        if cost_real == cost_predicted:
            return True, True
        else:
            return True, False
    else:
        return False, False

def test(dataloader, model, cost_matrix):
    model.eval()
    valid = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, z in dataloader:
            pred = model(x)
            y = y.view(-1, 12)
            pred_resize = np.array([1.0 if l > 0.5 else 0.0 for i in pred for l in i]).reshape((12,12))
            is_valid_path, is_shortest_path = same_path_cost(z.numpy()[0], pred_resize, y.numpy())
            if is_valid_path:
                valid += 1
            if is_shortest_path:
                correct += 1
            total += 1

    print("Valid paths ratio:", valid / total)
    print("Correct shortest path ratio:", correct / total)
    return correct / total
                
def train_and_test(model_file_name, train_set_x, train_set_y, val_set_x, val_set_y, test_set_x, 
                   test_set_y, nb_epochs, batch_size, learning_rate, use_dropout,
                   cost_matrix_train, cost_matrix_val, cost_matrix_test):
    if use_dropout:
        model = Net_NN_Dropout()
    else:
        model = Net_NN()
    sl = SemanticLoss('constraint.sdd', 'constraint.vtree')
    # loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_set_x = np.array([[cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)] for picture in train_set_x])
    val_set_x = np.array([[cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)] for picture in val_set_x])
    test_set_x = np.array([[cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)] for picture in test_set_x])

    # train_set_x = np.array([picture for picture in train_set_x])
    # val_set_x = np.array([picture for picture in val_set_x])
    # test_set_x = np.array([picture for picture in test_set_x])

    # plt.imshow(train_set_x[0][0], cmap='gray')
    # plt.show()

    # plt.imshow(train_set_y[0], cmap='gray')
    # plt.show()

    train_dataset = TensorDataset(torch.FloatTensor(train_set_x), torch.FloatTensor(train_set_y))
    val_dataset = TensorDataset(torch.FloatTensor(val_set_x), torch.FloatTensor(val_set_y), torch.FloatTensor(cost_matrix_val))
    test_dataset = TensorDataset(torch.FloatTensor(test_set_x), torch.FloatTensor(test_set_y), torch.FloatTensor(cost_matrix_test))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=1)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    # training (with early stopping)
    total_training_time = 0
    best_accuracy = 0
    counter = 0
    for epoch in range(nb_epochs):
        start_time = time.time()
        train(train_dataloader, model, sl, loss_fn, optimizer)
        total_training_time += time.time() - start_time
        val_accuracy = test(val_dataloader, model, cost_matrix_val)
        print("Val accuracy after epoch", epoch, ":", val_accuracy)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            with open("best_model.pickle", "wb") as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            counter = 0
        else:
            if counter >= 100:
                break
            counter += 1
    with open("best_model.pickle", "rb") as handle:
        model = pickle.load(handle)

    os.remove("best_model.pickle")

    # save trained model to a file
    with open("results/final/{}".format(model_file_name), "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    # testing
    start_time = time.time()
    accuracy = test(test_dataloader, model, cost_matrix_test)
    testing_time = time.time() - start_time

    return accuracy, total_training_time, testing_time

############################################### PARAMETERS ##############################################
nb_epochs = 100
batch_size = 256
learning_rate = 0.001
use_dropout = False
size_val = 0.1
#########################################################################################################

for seed in range(0, 10):
    # setting seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # shuffle and import train, val and test set
    train_set_x, train_set_y, cost_matrix_train = generate_dataset("train")
    val_set_x, val_set_y, cost_matrix_val = generate_dataset("val")
    test_set_x, test_set_y, cost_matrix_test = generate_dataset("test")

    # generate name of file that holds the trained model
    model_file_name = "NN_final_{}_{}_{}_{}_{}_{}".format(seed, nb_epochs, batch_size, learning_rate, 
        use_dropout, size_val)

    # train and test
    accuracy, training_time, testing_time = train_and_test(model_file_name, train_set_x, train_set_y, 
        val_set_x, val_set_y, test_set_x, test_set_y, nb_epochs, batch_size, learning_rate, use_dropout,
        cost_matrix_train, cost_matrix_val, cost_matrix_test)
    
    # save results to a summary file
    information = {
        "algorithm": "NN",
        "seed": seed,
        "nb_epochs": nb_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "use_dropout": use_dropout,
        "size_val": size_val,
        "accuracy": accuracy,
        "training_time": training_time,
        "testing_time": testing_time,
        "model_file": model_file_name
    }
    with open("results/summary_final.json", "a") as outfile:
        json.dump(information, outfile)
        outfile.write('\n')

    # print results
    print("############################################")
    print("Seed: {} \nAccuracy: {} \nTraining time: {} \nTesting time: {}".format(seed, accuracy, 
        training_time, testing_time))
    print("############################################")