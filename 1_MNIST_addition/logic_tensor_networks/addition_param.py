import ltn
import sys
import random
import numpy
import torch
import baselines
import tensorflow as tf
from tensorflow.keras import layers
from collections import defaultdict
from import_data import get_mnist_op_dataset_k_fold
from commons import train_modified, test_modified
from itertools import product
from sklearn.model_selection import KFold

sys.path.append("..")
from data.generate_dataset import generate_dataset

def train_and_test(train_set, val_set, nb_epochs, learning_rate, p_schedule):
    # predicates
    logits_model = baselines.SingleDigit()
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
        predictions_x = tf.argmax(logits_model(images_x),axis=-1)
        predictions_y = tf.argmax(logits_model(images_y),axis=-1)
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

    scheduled_parameters = defaultdict(lambda: {})
    for epoch in range(0, nb_epochs):
        scheduled_parameters[epoch] = {"p_schedule":tf.constant(p_schedule)}

    # training
    training_time = train_modified(train_set, train_step, scheduled_parameters, nb_epochs)

    # testing
    accuracy = test_modified(val_set, test_step, metrics_dict, scheduled_parameters)

    return accuracy, training_time

############################################### PARAMETERS ##############################################
possible_nb_epochs = [3]
possible_batch_size = [32, 64, 128]
possible_learning_rate = [0.001]
possible_p_schedule = [1.]
k = 10
#########################################################################################################

for seed in range(9, 10):
    for param in product(possible_nb_epochs, possible_batch_size, possible_learning_rate, possible_p_schedule):
        nb_epochs, batch_size, learning_rate, p_schedule = param
        
        # setting seeds for reproducibility
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)

        # import train and test set (shuffled)
        generate_dataset(seed)
        train_data, train_labels = get_mnist_op_dataset_k_fold()
        accuracy = 0
        fold_nb = 1

        for train_index, test_index in KFold(k).split(train_labels):
            ds_train = [numpy.array(train_data[0])[train_index], numpy.array(train_data[1])[train_index]]
            ds_test = [numpy.array(train_data[0])[test_index], numpy.array(train_data[1])[test_index]]
            labels_train = numpy.array(train_labels)[train_index]
            labels_test = numpy.array(train_labels)[test_index]
            ds_train = tf.data.Dataset.from_tensor_slices(tuple(ds_train)+(labels_train,)).batch(batch_size)
            ds_test = tf.data.Dataset.from_tensor_slices(tuple(ds_test)+(labels_test,)).batch(1)
            fold_accuracy, _ = train_and_test(ds_train, ds_test, nb_epochs, learning_rate, p_schedule)
            print(fold_nb, "-- Fold accuracy: ", fold_accuracy)
            fold_nb += 1
            accuracy += fold_accuracy

        accuracy /= k

        # print results
        print("############################################")
        print("seed: {}".format(seed))
        print("nb_epochs: {}".format(nb_epochs))
        print("batch_size: {}".format(batch_size))
        print("learning_rate: {}".format(learning_rate))
        print("p_schedule: {}".format(p_schedule))
        print("----------")
        print("Accuracy: {}".format(accuracy))
        print("############################################")
