import numpy as np
from data.shortest_path_dijkstra import find_shortest_path
from sklearn.utils import shuffle

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
        dataset_cost_matrix = []

        # create new dataset
        for map_id in range(len(original_dataset["maps"])):
            cost = original_dataset["costs"][map_id]
            map = original_dataset["maps"][map_id]
            path = find_shortest_path(cost)
            dataset_x.append(map)
            dataset_cost_matrix.append(cost)
            dataset_y.append(path)

        np.save(f'Warcraft/processed/{dataset_name}_x', np.array(dataset_x))
        np.save(f'Warcraft/processed/{dataset_name}_y', np.array(dataset_y))
        np.save(f'Warcraft/processed/{dataset_name}_cost_matrix', np.array(dataset_cost_matrix))

# import the processed dataset
def generate_dataset(dataset_name):
    dataset_x = np.load(f'../data/Warcraft/processed/{dataset_name}_x.npy')
    dataset_y = np.load(f'../data/Warcraft/processed/{dataset_name}_y.npy')
    dataset_cost_matrix = np.load(f'../data/Warcraft/processed/{dataset_name}_cost_matrix.npy')
    dataset_x, dataset_y, dataset_cost_matrix = shuffle(dataset_x, dataset_y, dataset_cost_matrix)

    # dataset = []
    # for i in range(len(dataset_x)):
    #     # display_map_and_path(dataset_x[i], dataset_y[i])
    #     dataset.append((dataset_x[i], dataset_y[i]))

    return dataset_x, dataset_y, dataset_cost_matrix

# generate_and_save_processed_datasets()