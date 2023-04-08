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
from collections import defaultdict
from import_data import get_dataset
from commons import train_modified, test_modified

sys.path.append("..")
from data.network_tensorflow import Net, Net_Dropout

def train_and_test(model_file_name, train_set, val_set, test_set, nb_epochs, learning_rate, p_schedule, 
    use_dropout):

    with open('../data/CiteSeer/cites.txt') as f:
        lines = f.readlines()
    cites = []
    for line in lines:
        cite_1 = int(line.split(",")[0].split("(")[-1])
        cite_2 = int(line.split(",")[1].split(")")[0])
        cites.append((cite_1, cite_2))
    cite_set = tf.data.Dataset.from_tensor_slices(cites)

    # predicates
    if use_dropout:
        logits_model = Net_Dropout()
    else:
        logits_model = Net()
    Document_type = ltn.Predicate(ltn.utils.LogitsToPredicateModel(logits_model))

    # variables
    d1 = ltn.Variable("digits1", range(6))

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

    # # axioms
    # @tf.function
    # def axioms(images_x, images_y, labels_z, p_schedule=tf.constant(2.)):
    #     images_x = ltn.Variable("x", images_x)
    #     images_y = ltn.Variable("y", images_y)
    #     labels_z = ltn.Variable("z", labels_z)
    #     axiom = Forall(
    #             ltn.diag(images_x,images_y,labels_z),
    #             Exists(
    #                 (d1,d2),
    #                 And(Digit([images_x,d1]),Digit([images_y,d2])),
    #                 mask=equals([add([d1,d2]), labels_z]),
    #                 p=p_schedule
    #             ),
    #             p=2
    #         )
    #     sat = axiom.tensor
    #     return sat

    # axioms
    @tf.function
    def axioms(features_x, labels_z, p_schedule=tf.constant(2.)):
        features_x = ltn.Variable("x", features_x)
        labels_z = ltn.Variable("z", labels_z)
        axiom = Forall(
                ltn.diag(features_x,labels_z),
                Exists(
                    (d1),
                    Document_type([features_x,d1]),
                    mask=equals([d1, labels_z]),
                    p=p_schedule
                ),
                p=2
            )
        
        sat = axiom.tensor
        return sat

    # initialize all layers
    features_x, labels_z = next(train_set.as_numpy_iterator())
    axioms(features_x, labels_z)

    # training
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    metrics_dict = {
        'train_loss': tf.keras.metrics.Mean(name="train_loss"),
        'train_accuracy': tf.keras.metrics.Mean(name="train_accuracy"),
        'test_loss': tf.keras.metrics.Mean(name="test_loss"),
        'test_accuracy': tf.keras.metrics.Mean(name="test_accuracy")    
    }

    @tf.function
    def train_step(features_x, labels_z, **parameters):
        with tf.GradientTape() as tape:
            loss = 1.- axioms(features_x, labels_z, **parameters)
        gradients = tape.gradient(loss, logits_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, logits_model.trainable_variables))
       
    @tf.function
    def test_step(images_x, labels_z, **parameters):
        predictions_x = tf.argmax(logits_model(images_x, training=False),axis=-1)

        match = tf.equal(predictions_x,tf.cast(labels_z,predictions_x.dtype))
        metrics_dict['test_accuracy'](tf.reduce_mean(tf.cast(match,tf.float32)))

    # the parameter p_schedule is the same in every epoch
    scheduled_parameters = defaultdict(lambda: {})
    for epoch in range(0, nb_epochs):
        scheduled_parameters[epoch] = {"p_schedule":tf.constant(p_schedule)}

    # training (with early stopping)
    total_training_time = 0
    best_accuracy = 0
    counter = 0
    for epoch in range(nb_epochs):
        total_training_time += train_modified(train_set, train_step, scheduled_parameters, 1)
        val_accuracy = test_modified(val_set, test_step, metrics_dict, scheduled_parameters)
        print("Val accuracy after epoch", epoch, ":", val_accuracy)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            with open("best_model.pickle", "wb") as handle:
                pickle.dump(logits_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            counter = 0
        else:
            if counter >= 0:
                break
            counter += 1
    with open("best_model.pickle", "rb") as handle:
        logits_model = pickle.load(handle)

    os.remove("best_model.pickle")

    # save trained model to a file
    with open("results/final/{}".format(model_file_name), "wb") as handle:
        pickle.dump(logits_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    # testing
    start_time = time.time()
    accuracy = test_modified(test_set, test_step, metrics_dict, scheduled_parameters)
    testing_time = time.time() - start_time

    return accuracy, total_training_time, testing_time

############################################### PARAMETERS ##############################################
nb_epochs = 3
batch_size = 16
learning_rate = 0.001
p_schedule = 1.
use_dropout = True
#########################################################################################################

for seed in range(0, 10):
    # setting seeds for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    # import train, val and test set
    train_set = get_dataset("train", batch_size)
    val_set = get_dataset("val", 1)
    test_set = get_dataset("test", 1)

    # generate name of folder that holds all the trained models
    model_file_name = "LTN_final_{}_{}_{}_{}_{}_{}".format(seed, nb_epochs, batch_size, learning_rate, 
        p_schedule, use_dropout)

    # train and test
    accuracy, training_time, testing_time = train_and_test(model_file_name, train_set, val_set, test_set, 
        nb_epochs, learning_rate, p_schedule, use_dropout)
      
    # save results to a summary file
    information = {
        "algorithm": "LTN",
        "seed": seed,
        "nb_epochs": nb_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "p_schedule": p_schedule,
        "use_dropout": use_dropout,
        "accuracy": float(accuracy),
        "training_time": float(training_time),
        "testing_time": float(testing_time),
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