import ltn
import os
import sys
import random
import numpy
import torch
import json
import pickle
import tensorflow as tf
from tensorflow.keras import layers
from collections import defaultdict
from import_data import get_mnist_op_dataset_k_fold
from commons import train_modified, test_modified
from sklearn.model_selection import KFold

sys.path.append("..")
from data.generate_dataset import generate_dataset
from data.network_tensorflow import Net, Net_Dropout, Net_Original

def train_and_test(model_file_name_dir, fold_nb, train_set, test_set, nb_epochs, learning_rate, p_schedule, 
    use_dropout):
    # predicates
    if use_dropout:
        logits_model = Net_Dropout()
    else:
        logits_model = Net()
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
    def axioms(images_x, images_y, labels_z, p_schedule=tf.constant(2.)):
        images_x = ltn.Variable("x", images_x)
        images_y = ltn.Variable("y", images_y)
        labels_z = ltn.Variable("z", labels_z)
        axiom = Forall(
                ltn.diag(images_x,images_y,labels_z),
                Exists(
                    (d1,d2),
                    And(Digit([images_x,d1]),Digit([images_y,d2])),
                    mask=equals([add([d1,d2]), labels_z]),
                    p=p_schedule
                ),
                p=2
            )
        sat = axiom.tensor
        return sat

    # initialize all layers
    images_x, images_y, labels_z = next(train_set.as_numpy_iterator())
    axioms(images_x, images_y, labels_z)

    # training
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    metrics_dict = {
        'train_loss': tf.keras.metrics.Mean(name="train_loss"),
        'train_accuracy': tf.keras.metrics.Mean(name="train_accuracy"),
        'test_loss': tf.keras.metrics.Mean(name="test_loss"),
        'test_accuracy': tf.keras.metrics.Mean(name="test_accuracy")    
    }

    @tf.function
    def train_step(images_x, images_y, labels_z, **parameters):
        # loss
        with tf.GradientTape() as tape:
            loss = 1.- axioms(images_x, images_y, labels_z, **parameters)
        gradients = tape.gradient(loss, logits_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, logits_model.trainable_variables))
        metrics_dict['train_loss'](loss)
        # accuracy
        predictions_x = tf.argmax(logits_model(images_x, True),axis=-1)
        predictions_y = tf.argmax(logits_model(images_y, True),axis=-1)
        predictions_z = predictions_x + predictions_y
        match = tf.equal(predictions_z,tf.cast(labels_z,predictions_z.dtype))
        metrics_dict['train_accuracy'](tf.reduce_mean(tf.cast(match,tf.float32)))
        
    @tf.function
    def test_step(images_x, images_y, labels_z, **parameters):
        # loss
        loss = 1.- axioms(images_x, images_y, labels_z, **parameters)
        metrics_dict['test_loss'](loss)
        # accuracy
        predictions_x = tf.argmax(logits_model(images_x),axis=-1)
        predictions_y = tf.argmax(logits_model(images_y),axis=-1)
        predictions_z = predictions_x + predictions_y
        match = tf.equal(predictions_z,tf.cast(labels_z,predictions_z.dtype))
        metrics_dict['test_accuracy'](tf.reduce_mean(tf.cast(match,tf.float32)))

    # the parameter p_schedule is the same in every epoch
    scheduled_parameters = defaultdict(lambda: {})
    for epoch in range(0, nb_epochs):
        scheduled_parameters[epoch] = {"p_schedule":tf.constant(p_schedule)}

    # training
    train_modified(train_set, train_step, scheduled_parameters, nb_epochs)

    # save trained model to a file
    path = "results/param/{}".format(model_file_name_dir)
    if not os.path.exists(path):
        os.makedirs(path)
    with open("results/param/{}/fold_{}".format(model_file_name_dir, fold_nb), "wb+") as handle:
        pickle.dump(logits_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    # testing
    accuracy = test_modified(test_set, test_step, metrics_dict, scheduled_parameters)

    return accuracy

############################################### PARAMETERS ##############################################
seed = 0
nb_epochs = 1
batch_size = 128
learning_rate = 0.001
p_schedule = 1.
use_dropout = True
#########################################################################################################

# (2, 64, 0.001, False)
# (1, 16, 0.001, False)
# (2, 16, 0.001, True)
# (3, 16, 0.001, True)
# (2, 128, 0.001, False)
# (3, 128, 0.001, False)
# (2, 16, 0.001, False)
# (3, 32, 0.001, True)
# (1, 64, 0.001, False)

# setting seeds for reproducibility
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)

# import train and test set (shuffled)
generate_dataset(seed)
train_data, train_labels = get_mnist_op_dataset_k_fold()
accuracies = []
fold_nb = 1

# generate name of folder that holds all the trained models
model_file_name_dir = "LTN_param_{}_{}_{}_{}_{}_{}".format(seed, nb_epochs, batch_size, learning_rate, 
    p_schedule, use_dropout)

for train_index, test_index in KFold(10).split(train_labels):
    ds_train = [numpy.array(train_data[0])[train_index], numpy.array(train_data[1])[train_index]]
    ds_test = [numpy.array(train_data[0])[test_index], numpy.array(train_data[1])[test_index]]
    labels_train = numpy.array(train_labels)[train_index]
    labels_test = numpy.array(train_labels)[test_index]
    ds_train = tf.data.Dataset.from_tensor_slices(tuple(ds_train)+(labels_train,)).batch(batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices(tuple(ds_test)+(labels_test,)).batch(1)
    fold_accuracy = train_and_test(model_file_name_dir, fold_nb, ds_train, ds_test, nb_epochs, 
        learning_rate, p_schedule, use_dropout)
    print(fold_nb, "-- Fold accuracy: ", fold_accuracy)
    accuracies.append(float(fold_accuracy))
    fold_nb += 1

avg_accuracy = sum(accuracies) / 10

# save results to a summary file
information = {
    "algorithm": "LTN",
    "seed": seed,
    "nb_epochs": nb_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "p_schedule": p_schedule,
    "use_dropout": use_dropout,
    "accuracies": accuracies,
    "avg_accuracy": avg_accuracy,
    "model_files_dir": model_file_name_dir
}
with open("results/summary_param.json", "a") as outfile:
    json.dump(information, outfile)
    outfile.write('\n')

# print results
print("############################################")
print("Seed: {} \nAccuracy: {}".format(seed, avg_accuracy))
print("############################################")