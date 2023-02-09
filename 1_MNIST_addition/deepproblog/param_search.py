import random
from itertools import product

random.seed(33)

possible_nb_epochs = [1, 2, 3]
possible_batch_size = [2, 4, 8, 16, 32, 64]
possible_learning_rate = [0.001]
possible_dropout = [0, 1]

possible_combinations = []

for combination in product(possible_nb_epochs, possible_batch_size, possible_learning_rate, possible_dropout):
    possible_combinations.append(combination)

selection = random.sample(possible_combinations, 15)

for combination in selection:
    print(combination)