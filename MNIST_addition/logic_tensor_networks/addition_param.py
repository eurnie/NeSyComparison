import ltn
import sys
import random
import numpy
import torch
import pickle
import json
import tensorflow as tf
from tensorflow.keras import layers
from collections import defaultdict
from import_data import get_mnist_op_dataset_val
from commons import train_modified, test_modified

sys.path.append("..")
from data.generate_dataset import generate_dataset_mnist, generate_dataset_fashion_mnist
from data.network_tensorflow import Net, Net_Dropout, Net_Original

def train_and_test(dataset, model_file_name, train_set, val_set, nb_epochs, learning_rate, 
                   p_schedule, use_dropout):
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
        with tf.GradientTape() as tape:
            loss = 1.- axioms(images_x, images_y, labels_z, **parameters)
        gradients = tape.gradient(loss, logits_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, logits_model.trainable_variables))
        
    @tf.function
    def test_step(images_x, images_y, labels_z, **parameters):
        predictions_x = tf.argmax(logits_model(images_x, training=False),axis=-1)
        predictions_y = tf.argmax(logits_model(images_y, training=False),axis=-1)
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
    with open(f'results/{dataset}/{model_file_name}', "wb") as handle:
        pickle.dump(logits_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    # testing
    accuracy = test_modified(val_set, test_step, metrics_dict, scheduled_parameters)
    return accuracy

################################################# DATASET ###############################################
dataset = "mnist"
# dataset = "fashion_mnist"
#########################################################################################################

############################################### PARAMETERS ##############################################
seed = 0
nb_epochs = 10
batch_size = 4
learning_rate = 0.001
p_schedule = 1.
use_dropout = False
size_val = 0.1
#########################################################################################################

# setting seeds for reproducibility
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)

# generate and shuffle dataset
if dataset == "mnist":
    generate_dataset_mnist(seed, 0)
elif dataset == "fashion_mnist":
    generate_dataset_fashion_mnist(seed, 0)

# import train and val set
train_set, val_set, _ = get_mnist_op_dataset_val(dataset, size_val, batch_size)

# generate name of folder that holds all the trained models
model_file_name = "param/LTN_param_{}_{}_{}_{}_{}_{}_{}".format(seed, nb_epochs, batch_size, 
        learning_rate, p_schedule, use_dropout, size_val)

# train and test
accuracy = train_and_test(dataset, model_file_name, train_set, val_set, nb_epochs, learning_rate, 
                          p_schedule, use_dropout)
    
# save results to a summary file
information = {
    "algorithm": "LTN",
    "seed": seed,
    "nb_epochs": nb_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "p_schedule": p_schedule,
    "use_dropout": use_dropout,
    "size_val": size_val,
    "accuracy": float(accuracy),
    "model_file": model_file_name
}
with open(f'results/{dataset}/param/summary_param.json', "a") as outfile:
    json.dump(information, outfile)
    outfile.write('\n')

# print results
print("############################################")
print("Accuracy: {}".format(accuracy))
print("############################################")
