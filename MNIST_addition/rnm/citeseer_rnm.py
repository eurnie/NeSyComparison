import mme
import tensorflow as tf
import datasets
import numpy as np
import os
from itertools import product

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import scipy.ndimage as img
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
import torchvision
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path

DATA_ROOT = Path(__file__).parent.parent.joinpath('data')

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

datasets = {
    "train": torchvision.datasets.MNIST(
        root=str(DATA_ROOT), train=True, download=True, transform=transform
    ),
    "test": torchvision.datasets.MNIST(
        root=str(DATA_ROOT), train=False, download=True, transform=transform
    ),
}

def parse_data(filename):
    with open(filename) as f:
        entries = f.readlines()

    dataset = []
    labels = []

    for entry in entries:
        index_digit_1 = int(entry.split(" ")[0])
        index_digit_2 = int(entry.split(" ")[1])
        sum = int(entry.split(" ")[2])

        new_entry = []
        new_entry.append(index_digit_1)
        new_entry.append(index_digit_2)
        dataset.append(new_entry)

        labels.append(sum)
        
    return dataset, labels

def get_mnist_data_as_numpy():
    """Returns numpy arrays of images and labels"""
    img_train = datasets["train"].data.numpy()
    img_train = img_train[:, :, :, np.newaxis]
    img_train = img_train/255.0
    label_train = datasets["train"].targets.numpy()
    img_test = datasets["test"].data.numpy()
    img_test = img_test/255.0
    img_test = img_test[:, :, :, np.newaxis]
    label_test = datasets["test"].targets.numpy()
    
    return img_train, label_train, img_test, label_test

def get_mnist_op_dataset_val(size_val, batch_size):
    img_train, _, img_test, _ = get_mnist_data_as_numpy()
    train_data_processed, train_labels = parse_data("../data/MNIST/processed/train.txt")
    test_data_processed, label_result_test = parse_data("../data/MNIST/processed/test.txt")
    
    split_index = round(size_val * 30000)
    label_result_val = train_labels[:split_index]
    label_result_train = train_labels[split_index:]

    img_per_operand_val_1 = [img_train[train_data_processed[i][0]] for i in range(0, split_index)]
    img_per_operand_val_2 = [img_train[train_data_processed[i][1]] for i in range(0, split_index)]
    img_per_operand_val = [img_per_operand_val_1, img_per_operand_val_2]

    img_per_operand_train_1 = [img_train[train_data_processed[i][0]] for i in range(split_index, len(train_data_processed))]
    img_per_operand_train_2 = [img_train[train_data_processed[i][1]] for i in range(split_index, len(train_data_processed))]
    img_per_operand_train = [img_per_operand_train_1, img_per_operand_train_2]

    img_per_operand_test_1 = [img_test[i[0]] for i in test_data_processed]
    img_per_operand_test_2 = [img_test[i[1]] for i in test_data_processed]
    img_per_operand_test = [img_per_operand_test_1, img_per_operand_test_2]

    ds_train = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_train)+(label_result_train,)).batch(batch_size)
    ds_val = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_val)+(label_result_val,)).batch(1)
    ds_test = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_test)+(label_result_test,)).batch(1)

    return ds_train, ds_val, ds_test

tf.get_logger().setLevel('ERROR')

base_savings = os.path.join("savings", "citeseer")
pretrain_path = os.path.join(base_savings,"pretrain")
posttrain_path = os.path.join(base_savings,"posttrain")

def citeseer_em(test_size, valid_size, seed):
    documents = np.load("data/citeseer/words.npy")
    n = len(documents)
    documents = documents[:n]
    labels = np.load("data/citeseer/labels.npy")
    labels = labels[:n]
    labels = np.eye(6)[labels]
    citations = np.greater(np.load("data/citeseer/citations.npy"), 0).astype(np.float32)
    citations = citations[:n, :n]
    num_documents = len(documents)

    def _inner_take_hb(idx):
        x = documents[idx]
        l = np.reshape(labels[idx].T, [1, -1])
        c = np.reshape(citations[idx][:, idx], [1, -1])

        hb = np.concatenate((l, c), axis=1)
        hb = hb.astype(np.float32)

        return x, hb

    trid, teid = train_test_split(np.arange(num_documents), test_size=test_size, random_state=0)
    trid, vaid = train_test_split(trid, test_size=valid_size, random_state=seed) if valid_size>0 else (trid, None)

    mask_train_labels = np.zeros_like(labels)
    mask_train_labels[trid] = 1

    x_train, hb_train = _inner_take_hb(trid)
    x_valid, hb_valid = _inner_take_hb(vaid) if valid_size>0 else (None, None)
    x_test, hb_test = _inner_take_hb(teid)
    x_all, hb_all = _inner_take_hb(np.arange(n))

    return (x_train, hb_train), (x_valid, hb_valid), (x_test, hb_test), (x_all, hb_all), labels, mask_train_labels, trid, vaid, teid

def main(lr,lambda_0,l2w, run_on_test=False, map_steps = 20, em_cycles=4):
    test_size = 0.1
    valid_size = 0.1

    (x_train, hb_train), (x_valid, hb_valid), (x_test, hb_test), (x_all, hb_all), labels, mask_train_labels, trid, vaid, teid= citeseer_em(test_size, valid_size)
    num_examples = len(x_all)
    num_classes = 6

    # T because we made classes as unary potentials
    indices = np.reshape(np.arange(num_classes * len(x_all)), [num_classes, len(x_all)]).T 

    indices_train = indices[trid]
    if run_on_test:
        x_to_test = x_test
        hb_to_test = hb_test
        num_examples_to_test = len(x_test)
        indices_to_test = indices[teid]

    else:
        x_to_test = x_valid
        hb_to_test = hb_valid
        num_examples_to_test = len(x_valid)
        indices_valid = np.reshape(np.arange(num_classes * len(x_valid)),
                             [num_classes, len(x_valid)]).T  # T because we made classes as unary potentials
        indices_to_test = indices[vaid]

    y_to_test = tf.gather(hb_all[0], indices_to_test)

    """Logic Program Definition"""
    o = mme.Ontology()

    # Domains
    images = mme.Domain("Images", data=x_all)
    numbers = mme.Domain("Numbers", data=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).T)
    result = mme.Domain("Result", data=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]).T)
    o.add_domain([images, numbers, result])

    # Predicates
    digit = mme.Predicate("digit", domains=[images, numbers], given=True)
    sum = mme.Predicate("digit", domains=[images, images, result])
    o.add_predicate([digit, sum])

    """MME definition"""
    potentials = []
    # Supervision
    indices = np.reshape(np.arange(num_classes * docs.num_constants),
                         [num_classes, docs.num_constants]).T  # T because we made classes as unary potentials

    nn = tf.keras.Sequential()
    nn.add(tf.keras.layers.Input(shape=(x_train.shape[1],)))
    nn.add(tf.keras.layers.Dense(50, activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(l2w)))  # up to the last hidden layer
    nn.add(tf.keras.layers.Dense(50, activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(l2w)))  # up to the last hidden layer
    nn.add(tf.keras.layers.Dense(50, activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(l2w)))  # up to the last hidden layer
    nn.add(tf.keras.layers.Dense(num_classes, use_bias=False))
    p1 = mme.potentials.SupervisionLogicalPotential(nn, indices)
    potentials.append(p1)

    # Mutual Exclusivity (needed for inference , since SupervisionLogicalPotential already subsumes it during training)
    # p2 = mme.potentials.MutualExclusivityPotential(indices=indices)
    # potentials.append(p2)

    # Logical
    np.ones_like(hb_all)
    evidence_mask = np.zeros_like(hb_all)
    evidence_mask[:, num_examples * num_classes:]=1
    for name in preds:
        c = mme.Formula(definition="%s(x) and cite(x,y) -> %s(y)" % (name, name), ontology=o)
        p3 = mme.potentials.EvidenceLogicPotential(formula=c, logic=mme.logic.BooleanLogic, evidence=hb_all,
                                                   evidence_mask=evidence_mask)
        potentials.append(p3)

    P = mme.potentials.GlobalPotential(potentials)

    def pretrain_step():
        """pretrain rete"""
        y_train = tf.gather(hb_all[0], indices_train)

        adam = tf.keras.optimizers.Adam(lr=0.001)

        def training_step():
            with tf.GradientTape() as tape:
                neural_logits = nn(x_train)

                total_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=y_train,
                                                            logits=neural_logits)) + tf.reduce_sum(nn.losses)

            grads = tape.gradient(target=total_loss, sources=nn.variables)
            grad_vars = zip(grads, nn.variables)
            adam.apply_gradients(grad_vars)

        epochs_pretrain = 200
        for e in range(epochs_pretrain):
            training_step()
            y_nn = nn(x_to_test)
            acc_nn = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(y_to_test, axis=1), tf.argmax(y_nn, axis=1)), tf.float32))
            print(acc_nn)

        y_new = tf.gather(tf.eye(num_classes), tf.argmax(nn(x_all), axis=1), axis=0)

        new_labels = tf.where(mask_train_labels > 0, labels, y_new)
        new_hb = tf.concat(
            (tf.reshape(tf.transpose(new_labels, [1, 0]), [1, -1]), hb_all[:, num_examples * num_classes:]),axis=1)

        return new_hb

    def em_step(new_hb):

        pwt = mme.PieceWiseTraining(global_potential=P)
        hb = new_hb

        """BETA TRAINING"""
        pwt.compute_beta_logical_potentials(y=hb)
        for p in potentials:
            print(p, p.beta)

        """NN TRAINING"""
        epochs = 50

        for _ in range(epochs):
            pwt.maximize_likelihood_step(new_hb, x=x_all)
            y_nn = nn(x_to_test)
            acc_nn = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(y_to_test, axis=1), tf.argmax(y_nn, axis=1)), tf.float32))
            print("TRAINING", acc_nn.numpy())

        """Fix the training hb for inference"""
        new_labels = tf.where(mask_train_labels > 0, labels, 0.5*tf.ones_like(labels))
        evidence = tf.concat(
            (tf.reshape(tf.transpose(new_labels, [1, 0]), [1, -1]), hb_all[:, num_examples * num_classes:]), axis=1)
        evidence = tf.cast(evidence, tf.float32)
        evidence_mask = tf.concat(
            (tf.reshape(tf.transpose(mask_train_labels.astype(np.float32), [1, 0]), [1, -1]), tf.ones_like(hb_all[:, num_examples * num_classes:])), axis=1)>0

        """MAP Inference"""
        dict = {"var": tf.Variable(initial_value=nn(x_all)),
                "labels": labels,
                "mask_train_labels": mask_train_labels,
                "hb_all": hb_all,
                "num_examples": num_examples,
                "num_classes": num_classes}

        steps_map = map_steps
        map_inference = mme.inference.FuzzyMAPInference(y_shape=hb.shape,
                                                        potential=P,
                                                        logic=mme.logic.LukasiewiczLogic,
                                                        evidence=evidence,
                                                        evidence_mask=evidence_mask,
                                                        learning_rate=lr,
                                                        external_map = dict)

        P.potentials[0].beta = lambda_0
        y_map = tf.gather(map_inference.map()[0], indices_to_test)
        acc_map = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(y_to_test, axis=1), tf.argmax(y_map, axis=1)), tf.float32))
        print("MAP", acc_map.numpy())
        for i in range(steps_map):
            map_inference.infer_step(x_all)
            y_map = tf.gather(map_inference.map()[0], indices_to_test)
            acc_map = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(y_to_test, axis=1), tf.argmax(y_map, axis=1)), tf.float32))
            print("MAP", i, acc_map.numpy())
            if mme.utils.heardEnter():
                break

        new_hb = tf.cast(map_inference.map()>0.5, tf.float32)
        y_new_fuzzy = tf.gather(map_inference.map()[0], indices)
        new_labels = tf.where(mask_train_labels > 0, labels, y_new_fuzzy)
        new_hb_fuzzy = tf.concat(
            (tf.reshape(tf.transpose(new_labels, [1, 0]), [1, -1]), hb_all[:, num_examples * num_classes:]), axis=1)

        return new_hb, new_hb_fuzzy

    em_cycles = em_cycles
    for i in range(em_cycles):
        if i == 0:
            new_hb = hb_pretrain = pretrain_step()
        else:
            old_hb = new_hb
            new_hb, new_hb_fuzzy = em_step(old_hb)

    y_map = tf.gather(new_hb[0], indices_to_test)
    y_pretrain = tf.gather(hb_pretrain[0], indices_to_test)
    acc_pretrain = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(y_to_test, axis=1), tf.argmax(y_pretrain, axis=1)), tf.float32))
    acc_map = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(y_to_test, axis=1), tf.argmax(y_map, axis=1)), tf.float32))
    return acc_pretrain, acc_map

if __name__ == "__main__":
    res = []
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    lambda_0 = 0.001
    em_cycles = 4
    steps = 30

    acc_map, acc_nn = main(lr=1., lambda_0=lambda_0, l2w=0.006, run_on_test=True, map_steps=steps, em_cycles=em_cycles)
    acc_map, acc_nn = acc_map.numpy(), acc_nn.numpy()
    print("acc_map:", acc_map)
    print("acc_nn:", acc_nn)