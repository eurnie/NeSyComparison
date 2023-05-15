import random
import tensorflow as tf
import torch_geometric
from pathlib import Path

# import the dataset, shuffle it with the seed and move examples from the train set to the unsupervised setting
def import_datasets(dataset, to_unsupervised, batch_size, seed):
    DATA_ROOT = Path(__file__).parent.parent.joinpath('data')
    data = torch_geometric.datasets.Planetoid(root=str(DATA_ROOT), name=dataset, split="full")
    citation_graph = data[0]

    # create the train, val and test set
    for dataset_name in ["train", "val", "test"]:
        if (dataset_name == "train"):
            mask = citation_graph.train_mask
        elif (dataset_name == "val"):
            mask = citation_graph.val_mask
        elif (dataset_name == "test"):
            mask = citation_graph.test_mask

        x = citation_graph.x[mask]
        y = citation_graph.y[mask]

        # shuffle dataset
        dataset = [(x[i].numpy(), y[i].numpy()) for i in range(len(x))]
        rng = random.Random(seed)
        rng.shuffle(dataset)

        # move train examples to the test set according to the given ratio
        if dataset_name == "train":
            if to_unsupervised > 0:
                split_index = round(to_unsupervised * len(dataset))
                train_set = dataset[split_index:]
            else:
                train_set = dataset
        elif dataset_name == "val":
            val_set = dataset
        elif dataset_name == "test":
            test_set = dataset

    print("The training set contains", len(train_set), "instances.")
    print("The validation set contains", len(val_set), "instances.")
    print("The testing set contains", len(test_set), "instances.")

    x_train, y_train = zip(*train_set)
    train_set = tf.data.Dataset.from_tensor_slices((list(x_train), list(y_train))).batch(batch_size)
    x_val, y_val = zip(*val_set)
    val_set = tf.data.Dataset.from_tensor_slices((list(x_val), list(y_val))).batch(batch_size)
    x_test, y_test = zip(*test_set)
    test_set = tf.data.Dataset.from_tensor_slices((list(x_test), list(y_test))).batch(batch_size)

    # collect the cites
    list_1 = citation_graph.edge_index[0]
    list_2 = citation_graph.edge_index[1]

    cites_a = [citation_graph.x[list_1[i]] for i in range(len(list_1))]
    cites_b = [citation_graph.x[list_2[i]] for i in range(len(list_2))]

    return train_set, val_set, test_set, cites_a, cites_b