import ltn
import sys
import random
import numpy
import torch
import pickle
import time
import json
import os
import tensorflow as tf
from tensorflow.keras import layers
from import_data import get_mnist_op_dataset_val
from commons import train_modified, test_modified

sys.path.append("..")
from data.generate_dataset import generate_dataset_mnist, generate_dataset_fashion_mnist
from data.network_tensorflow import Net

def train_and_test(dataset, model_file_name, train_set, val_set, test_set, nb_epochs, learning_rate, 
                   p_forall, p_exists, dropout_rate):
    # predicates
    logits_model = Net(dropout_rate)
    Digit = ltn.Predicate(ltn.utils.LogitsToPredicateModel(logits_model))

    # variables
    d1 = ltn.Variable("digits1", range(10))
    d2 = ltn.Variable("digits2", range(10))

    # operators
    Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
    And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
    Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
    Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
    Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(),semantics="forall")
    Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(),semantics="exists")

    # mask
    add = ltn.Function.Lambda(lambda inputs: inputs[0]+inputs[1])
    equals = ltn.Predicate.Lambda(lambda inputs: inputs[0] == inputs[1])

    # axioms
    @tf.function
    def axioms(images_x, images_y, labels_z, p_forall, p_exists):
        images_x = ltn.Variable("x", images_x)
        images_y = ltn.Variable("y", images_y)
        labels_z = ltn.Variable("z", labels_z)
        axiom = Forall(
                ltn.diag(images_x,images_y,labels_z),
                Exists(
                    (d1,d2),
                    And(Digit([images_x,d1]),Digit([images_y,d2])),
                    mask=equals([add([d1,d2]), labels_z]),
                    p=p_exists
                ),
                p=p_forall
            )
        sat = axiom.tensor
        return sat

    # initialize all layers
    images_x, images_y, labels_z = next(train_set.as_numpy_iterator())
    axioms(images_x, images_y, labels_z, p_forall, p_exists)

    # training
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    metrics_dict = {
        'train_loss': tf.keras.metrics.Mean(name="train_loss"),
        'train_accuracy': tf.keras.metrics.Mean(name="train_accuracy"),
        'test_loss': tf.keras.metrics.Mean(name="test_loss"),
        'test_accuracy': tf.keras.metrics.Mean(name="test_accuracy")    
    }

    @tf.function
    def train_step(images_x, images_y, labels_z, p_forall, p_exists):
        with tf.GradientTape() as tape:
            loss = 1.- axioms(images_x, images_y, labels_z, p_forall, p_exists)
        gradients = tape.gradient(loss, logits_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, logits_model.trainable_variables))
        
    @tf.function
    def test_step(images_x, images_y, labels_z):
        predictions_x = tf.argmax(logits_model(images_x, training=False),axis=-1)
        predictions_y = tf.argmax(logits_model(images_y, training=False),axis=-1)
        predictions_z = predictions_x + predictions_y
        match = tf.equal(predictions_z,tf.cast(labels_z,predictions_z.dtype))
        metrics_dict['test_accuracy'](tf.reduce_mean(tf.cast(match,tf.float32)))

    # training (with early stopping)
    total_training_time = 0
    best_accuracy = 0
    counter = 0
    for epoch in range(nb_epochs):
        total_training_time += train_modified(train_set, train_step, p_forall, p_exists, 1)
        val_accuracy = test_modified(val_set, test_step, metrics_dict)
        print("Val accuracy after epoch", epoch, ":", val_accuracy)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            nb_epochs_done = epoch + 1
            with open("best_model.pickle", "wb") as handle:
                pickle.dump(logits_model.classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
            counter = 0
        else:
            if counter >= 2:
                break
            counter += 1

    # load best model
    logits_model = Net(0)
    with open("best_model.pickle", "rb") as handle:
        logits_model.classifier = pickle.load(handle)
    os.remove("best_model.pickle")

    # save trained model to a file
    with open(f'results/{dataset}/final/{model_file_name}', "wb") as handle:
        pickle.dump(logits_model.classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    # testing
    start_time = time.time()
    accuracy = test_modified(test_set, test_step, metrics_dict)
    testing_time = time.time() - start_time

    return nb_epochs_done, accuracy, total_training_time, testing_time

################################################# DATASET ###############################################
dataset = "mnist"
# dataset = "fashion_mnist"
label_noise = 0
#########################################################################################################

############################################### PARAMETERS ##############################################
nb_epochs = 100
batch_size = 64
learning_rate = 0.001
p_forall = 2
p_exists = 1
dropout_rate = 0
size_val = 0.1
#########################################################################################################

for seed in range(0, 10):
    # setting seeds for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)

    # generate and shuffle dataset
    if dataset == "mnist":
        generate_dataset_mnist(seed, label_noise)
    elif dataset == "fashion_mnist":
        generate_dataset_fashion_mnist(seed, label_noise)

    # import train, val and test set
    train_set, val_set, test_set = get_mnist_op_dataset_val(dataset, size_val, batch_size)

    # generate name of folder that holds all the trained models
    model_file_name = "label_noise_{}/LTN_final_{}_{}_{}_{}_{}_{}_{}_{}".format(label_noise, seed, 
        nb_epochs, batch_size, learning_rate, p_forall, p_exists, dropout_rate, size_val)

    # train and test
    nb_epochs_done, accuracy, training_time, testing_time = train_and_test(dataset, model_file_name, 
        train_set, val_set, test_set, nb_epochs, learning_rate, p_forall, p_exists, dropout_rate)
      
    # save results to a summary file
    information = {
        "algorithm": "LTN",
        "seed": seed,
        "nb_epochs": nb_epochs_done,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "p_forall": p_forall,
        "p_exists": p_exists,
        "dropout_rate": dropout_rate,
        "size_val": size_val,
        "accuracy": float(accuracy),
        "training_time": float(training_time),
        "testing_time": float(testing_time),
        "model_file": model_file_name
    }
    with open(f'results/{dataset}/final/label_noise_{label_noise}/summary_final.json', "a") as outfile:
        json.dump(information, outfile)
        outfile.write('\n')

    # print results
    print("############################################")
    print("Seed: {} \nAccuracy: {} \nTraining time: {} \nTesting time: {}".format(seed, accuracy, 
        training_time, testing_time))
    print("############################################")
