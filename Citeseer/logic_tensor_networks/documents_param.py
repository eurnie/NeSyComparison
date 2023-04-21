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
from import_data import get_dataset, get_cites
from commons import train_modified, test_modified

sys.path.append("..")
from data.network_tensorflow import Net, Net_Dropout

############################################### PARAMETERS ##############################################
seed = 0
nb_epochs = 100
batch_size = 64
learning_rate = 0.001
p_schedule = 1.
use_dropout = False
#########################################################################################################

# setting seeds for reproducibility
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)

# import train, val and test set
train_set = get_dataset("train", batch_size)
val_set = get_dataset("val", 1)
test_set = get_dataset("test", 1)
cite_a, cite_b = get_cites()

# predicates
if use_dropout:
    logits_model = Net_Dropout()
else:
    logits_model = Net()
Document_type = ltn.Predicate(ltn.utils.LogitsToPredicateModel(logits_model))

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

# variables
l1 = ltn.Variable("label1", range(6))
l2 = ltn.Variable("label2", range(6))

formula_aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError(p=5))

# axioms
@tf.function
def axioms(cite_a, cite_b, features_x, labels_z, p_schedule=tf.constant(2.)):
    features_x = ltn.Variable("x", features_x)
    labels_z = ltn.Variable("z", labels_z)
    cite_a = ltn.Variable("ca", cite_a)
    cite_b = ltn.Variable("cb", cite_b)

    axioms = []

    # 
    axioms.append(Forall(
            ltn.diag(features_x,labels_z),
            Exists(
                (l1),
                Document_type([features_x,l1]),
                mask=equals([l1, labels_z])
            )
        )
    )

    axioms.append(Forall(
                        ltn.diag(cite_a, cite_b),
                        Exists(
                                (l2),
                                And(Document_type([cite_a,l2]),Document_type([cite_b,l2]))
                                )
                        )
                    )
    
    return formula_aggregator(axioms).tensor

# initialize all layers
features_x, labels_z = next(train_set.as_numpy_iterator())
axioms(cite_a, cite_b, features_x, labels_z)

# training
optimizer = tf.keras.optimizers.Adam(learning_rate)
metrics_dict = {
    'train_loss': tf.keras.metrics.Mean(name="train_loss"),
    'train_accuracy': tf.keras.metrics.Mean(name="train_accuracy"),
    'test_loss': tf.keras.metrics.Mean(name="test_loss"),
    'test_accuracy': tf.keras.metrics.Mean(name="test_accuracy")    
}

@tf.function
def train_step(cite_a, cite_b, features_x, labels_z, **parameters):
    with tf.GradientTape() as tape:
        loss = 1.- axioms(cite_a, cite_b, features_x, labels_z, **parameters)
    gradients = tape.gradient(loss, logits_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, logits_model.trainable_variables))
    
@tf.function
def test_step(images_x, labels_z, **parameters):
    predictions_x = tf.argmax(logits_model(images_x, training=False),axis=-1)

    match = tf.equal(predictions_x,tf.cast(labels_z,predictions_x.dtype))
    metrics_dict['test_accuracy'](tf.reduce_mean(tf.cast(match,tf.float32)))

# the parameter p_schedule is the same in every epoch
scheduled_parameters = defaultdict(lambda: {})
# for epoch in range(0, nb_epochs):
#     scheduled_parameters[epoch] = {"p_schedule":tf.constant(p_schedule)}

best_accuracy = 0

# training
for epoch in range(nb_epochs):
    train_modified(cite_a, cite_b, train_set, train_step, scheduled_parameters, 1)

    # generate name of folder that holds all the trained models
    model_file_name = "LTN_param_{}_{}_{}_{}_{}_{}".format(seed, epoch + 1, batch_size, learning_rate, 
        p_schedule, use_dropout)


    # save trained model to a file
    with open("results/param/{}".format(model_file_name), "wb") as handle:
        pickle.dump(logits_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    # testing
    accuracy = test_modified(val_set, test_step, metrics_dict, scheduled_parameters)
        
    # save results to a summary file
    information = {
        "algorithm": "LTN",
        "seed": seed,
        "nb_epochs": epoch + 1,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "p_schedule": p_schedule,
        "use_dropout": use_dropout,
        "accuracy": float(accuracy),
        "model_file": model_file_name
    }
    with open("results/summary_param.json", "a") as outfile:
        json.dump(information, outfile)
        outfile.write('\n')

    # print results
    print("############################################")
    print("Accuracy: {}".format(accuracy))
    print("############################################")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        counter = 0
    else:
        if counter >= 2:
            break
        counter += 1