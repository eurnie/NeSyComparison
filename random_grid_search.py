import random
from itertools import product

random.seed(0)
TOTAL_COMBINATIONS = 12

########################### OLD ###########################

# deepproblog
# possible_nb_epochs = [1, 2, 3]
# possible_batch_size = [2, 4, 8]
# possible_learning_rate = [0.001]
# possible_dropout = [True, False]

# deepstochlog
# possible_nb_epochs = [1, 2, 3]
# possible_batch_size = [2, 4, 8]
# possible_learning_rate = [0.001]
# possible_dropout = [True, False]

# neurasp
# possible_nb_epochs = [1, 2, 3]
# possible_batch_size = [2, 4, 8, 16, 32]
# possible_learning_rate = [0.001]
# possible_dropout = [True, False]

# logic tensor networks
# possible_nb_epochs = [1, 2, 3]
# possible_batch_size = [16, 32, 64, 128]
# possible_learning_rate = [0.001]
# possible_dropout = [True, False]

# neural net
# possible_nb_epochs = [5, 10, 15]
# possible_batch_size = [2, 4, 8, 16, 32, 64, 128]
# possible_learning_rate = [0.001]
# possible_dropout = [True, False]

# semantic loss
# possible_nb_epochs = [1, 2, 3, 4, 5]
# possible_batch_size = [2, 4, 8, 16, 32]
# possible_learning_rate = [0.001]
# possible_dropout = [True, False]

########################### NEW ###########################

# deepproblog, deepstochlog, neurasp
possible_nb_epochs = [1, 2, 3]
possible_batch_size = [2, 4, 8, 16]
possible_learning_rate = [0.001]
possible_dropout = [False]

possible_combinations = []

for combination in product(possible_nb_epochs, possible_batch_size, possible_learning_rate, possible_dropout):
    possible_combinations.append(combination)

selection = random.sample(possible_combinations, TOTAL_COMBINATIONS)

for combination in selection:
    print(combination)