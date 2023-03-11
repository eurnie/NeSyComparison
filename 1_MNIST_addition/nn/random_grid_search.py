import random
from itertools import product

random.seed(0)
TOTAL_COMBINATIONS = 10

possible_nb_epochs = [10, 20, 30, 40, 50]
possible_batch_size = [2, 4, 8, 16, 32]
possible_learning_rate = [0.001]
possible_dropout = [True, False]

possible_combinations = []

for combination in product(possible_nb_epochs, possible_batch_size, possible_learning_rate, possible_dropout):
    possible_combinations.append(combination)

selection = random.sample(possible_combinations, TOTAL_COMBINATIONS)

for combination in selection:
    print(combination)