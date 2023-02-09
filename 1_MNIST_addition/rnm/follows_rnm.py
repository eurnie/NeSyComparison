import mme
import random
import torch
import tensorflow as tf
import import_data
import numpy as np
import os
import sys

sys.path.append("..")
from data.generate_dataset import generate_dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.get_logger().setLevel('ERROR')

def train_and_test(train_set, test_set, max_nb_epochs, learning_rate):
    x_train, train_indices, y_train = train_set
    x_test, test_indices, y_test = test_set


    # x_train, hb_train = train_set
    # x_test, hb_test = test_set

    # m_e = np.zeros_like(hb_train)
    # m_e[:, num_examples*10:] = 1

    # y_e_train = hb_train * m_e
    # y_e_test = hb_test * m_e

    """Logic Program Definition"""
    o = mme.Ontology()

    # domains
    images = mme.Domain("Images", data=x_train)
    numbers = mme.Domain("Numbers", data=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).T)
    o.add_domain([images, numbers])

    # predicates
    digit = mme.Predicate("digit", domains=[images, numbers])
    # links = mme.Predicate("links", domains=[images, images], given=True)
    # follows = mme.Predicate("follows", domains=[numbers, numbers], given=True)
    sum = mme.Predicate("sum", domains=[images, images, numbers], given = True)
    # o.add_predicate([digit, links, follows, sum])
    o.add_predicate([digit, sum])

    """MME definition"""
    # supervision
    indices = np.reshape(np.arange(images.num_constants * numbers.num_constants),
                         [images.num_constants, numbers.num_constants])
    nn = tf.keras.Sequential()
    nn.add(tf.keras.layers.Input(shape=(784,)))
    nn.add(tf.keras.layers.Dense(100, activation=tf.nn.sigmoid))  # up to the last hidden layer
    nn.add(tf.keras.layers.Dense(10,use_bias=False))
    p1 = mme.potentials.SupervisionLogicalPotential(nn, indices)

    # mutual Exclusivity (needed for inference , since SupervisionLogicalPotential already subsumes it during training)
    p2 = mme.potentials.MutualExclusivityPotential(indices=indices)

    # logical
    # c = mme.Formula(definition="links(x,y) and digit(x,i) and digit(y,j) -> follows(i,j)", ontology=o)
    c = mme.Formula(definition="digit(x,i) and digit(y,j) -> sum(x, y, i+j)", ontology=o)
    p3 = mme.potentials.EvidenceLogicPotential(formula=c,
                                               logic=mme.logic.BooleanLogic,
                                               evidence=y_e_train,
                                               evidence_mask=m_e)
    P = mme.potentials.GlobalPotential([p1,p2,p3])

    pwt = mme.PieceWiseTraining(global_potential=P, y=hb_train)
    pwt.compute_beta_logical_potentials()

    y_test = tf.reshape(hb_test[0, :num_examples * 10], [num_examples, 10])
    for _ in range(max_nb_epochs):
        pwt.maximize_likelihood_step(hb_train, x=x_train)
        y_nn = p1.model(x_test)
        acc_nn = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_nn, axis=1)), tf.float32))
        # print(acc_nn)

    """Inference"""
    steps_map = 500
    hb = hb_test
    x = x_test
    evidence = y_e_test
    evidence_mask = m_e>0

    P.potentials[0].beta = 0.01
    map_inference = mme.inference.FuzzyMAPInference(y_shape=hb.shape,
                                                    potential=P,
                                                    logic=mme.logic.LukasiewiczLogic,
                                                    evidence=evidence,
                                                    evidence_mask=evidence_mask,
                                                    learning_rate= learning_rate) #tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=steps_map, decay_rate=0.96, staircase=True))

    y_test = tf.reshape(hb[0, :num_examples * 10], [num_examples, 10])
    for i in range(steps_map):
        map_inference.infer_step(x)
        if i % 10 == 0:
            y_map = tf.reshape(map_inference.map()[0, :num_examples * 10], [num_examples, 10])
            acc_map = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_map, axis=1)), tf.float32))
            # print("Accuracy MAP", acc_map.numpy())
            # print(y_map[:3])
        if mme.utils.heardEnter():
            break

    y_map = tf.reshape(map_inference.map()[0, :num_examples * 10], [num_examples, 10])
    y_nn = p1.model(x)

    acc_map = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_map, axis=1)), tf.float32))
    acc_nn = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, axis=1), tf.argmax(y_nn, axis=1)), tf.float32))

    return acc_map.numpy(), acc_nn.numpy()

############################################### PARAMETERS ##############################################
max_nb_epochs = 150
# batch_size = 2
learning_rate = 0.1
#########################################################################################################

seed = 0
# for seed in range(0, 10):
# setting seeds for reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
tf.random.set_seed(seed)

# import train and test set (shuffled)
# generate_dataset(seed)
train_set, test_set = import_data.mnist_data()

# train and test the method on the MNIST addition dataset
accuracy, training_time = train_and_test(train_set, test_set, max_nb_epochs, learning_rate)

# print results
print("############################################")
print("Seed: {} \nAccuracy: {} \nTraining time: {}".format(seed, accuracy, training_time))
print("############################################")