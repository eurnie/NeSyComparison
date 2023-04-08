import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from shortest_path_dijkstra import find_shortest_path

# import the raw warcraft dataset
def load_npz(path_to_file):
    loaded = np.load(path_to_file)
    return {
        "maps": loaded["maps"],
        "costs": loaded["costs"],
        "targets": loaded["targets"],
        "sources": loaded["sources"],
        "paths": loaded["paths"],
        "exp_nodes": loaded["exp_nodes"],
    }

# display the given image of the map
def display_map(map):
    plt.imshow(map)
    plt.show()

# display the given image of the map and highlight the associated path
# path has a given start point (source) and end point (target)
def display_map_and_path(map, path, source, target):
    map_to_show = deepcopy(map)

    # highlight the given path
    for i in range(len(path)):
        for j in range(len(path[0])):
            if path[i][j] == 1:
                for l in range(8):
                    for k in range(8):
                        map_to_show[(i*8)+l][(j*8)+k] = 0

    # highlight the source location
    for l in range(8):
        for k in range(8):
            map_to_show[(int(source[0])*8)+l][(int(source[1])*8)+k] = [255, 0, 0]

    # highlight the target location
    for l in range(8):
        for k in range(8):
            map_to_show[(int(target[0])*8)+l][(int(target[1])*8)+k] = [0, 128, 0]

    plt.imshow(map_to_show)
    plt.show()

# display the given image of the map and highlight the associated path
# path starts in the left upper corner and ends in the right lower corner
def display_map_and_path(map, path):
    map_to_show = deepcopy(map)

    # highlight the given path
    for i in range(len(path)):
        for j in range(len(path[0])):
            if path[i][j] == 1:
                for l in range(8):
                    for k in range(8):
                        map_to_show[(i*8)+l][(j*8)+k] = 0

    plt.imshow(map_to_show)
    plt.show()

# import the raw dataset and save the processed dataset
def generate_and_save_processed_datasets():
    for dataset_name in ["train", "val", "test"]:
        # import original dataset
        if dataset_name == "train":
            original_dataset = load_npz("Warcraft/raw/train.npz")
        elif dataset_name == "val":
            original_dataset = load_npz("Warcraft/raw/val.npz")
        elif dataset_name == "test":
            original_dataset = load_npz("Warcraft/raw/test.npz")

        dataset_x = []
        dataset_y = []

        # create new dataset
        for map_id in range(len(original_dataset["maps"])):
            cost = original_dataset["costs"][map_id]
            map = original_dataset["maps"][map_id]
            path = find_shortest_path(cost)
            dataset_x.append(map)
            dataset_y.append(path)

        np.save(f'Warcraft/processed/{dataset_name}_x', np.array(dataset_x))
        np.save(f'Warcraft/processed/{dataset_name}_y', np.array(dataset_y))

# import the processed dataset
def generate_dataset(dataset_name, seed):
    dataset_x = np.load(f'Warcraft/processed/{dataset_name}_x.npy')
    dataset_y = np.load(f'Warcraft/processed/{dataset_name}_y.npy')

    dataset = []

    for i in range(len(dataset_x)):
        # display_map_and_path(dataset_x[i], dataset_y[i])
        dataset.append((dataset_x[i], dataset_y[i]))

    rng = random.Random(seed)
    rng.shuffle(dataset)

    return dataset

generate_and_save_processed_datasets()
train_set = generate_dataset("train", 0)
val_set = generate_dataset("val", 0)
test_set = generate_dataset("test", 0)