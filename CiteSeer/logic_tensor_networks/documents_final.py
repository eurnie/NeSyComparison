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
from import_data import import_datasets
from commons import train_modified, test_modified

sys.path.append("..")
from data.network_tensorflow import Net_CiteSeer, Net_Cora, Net_PubMed

############################################### PARAMETERS ##############################################
nb_epochs = 100
batch_size = 512
learning_rate = 0.001
p_forall = 2
p_exists = 1
p_forall_cites = 2
p_exists_cites = 1
dropout_rate = 0
#########################################################################################################

for dataset, to_unsupervised in [("Cora", 0), ("CiteSeer", 0.1), ("CiteSeer", 0.25), ("CiteSeer", 0.5)]:
# for dataset, to_unsupervised in [("CiteSeer", 0)]:
    assert (dataset == "CiteSeer") or (dataset == "Cora") or (dataset == "PubMed")
    for seed in range(0, 10):
        # generate name of file that holds the trained model
        model_file_name = "LTN_final_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(seed, 
            nb_epochs, batch_size, learning_rate, p_forall, p_exists, p_forall_cites,
            p_exists_cites, dropout_rate)
        model_file_location = f'results/{dataset}/final/to_unsupervised_{to_unsupervised}/{model_file_name}'

        if not os.path.isfile(model_file_location):
            # setting seeds for reproducibility
            random.seed(seed)
            numpy.random.seed(seed)
            torch.manual_seed(seed)
            tf.random.set_seed(seed)

            # import train, val and test set + the citation network
            train_set, val_set, test_set, cites_a, cites_b = import_datasets(dataset, to_unsupervised, batch_size, seed)

            # predicates
            if dataset == "CiteSeer":
                logits_model = Net_CiteSeer(dropout_rate)
            elif dataset == "Cora":
                logits_model = Net_Cora(dropout_rate)
            elif dataset == "PubMed":
                logits_model = Net_PubMed(dropout_rate)
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

            # variables
            l1 = ltn.Variable("label1", range(6))
            formula_aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError(p=5))

            # axioms
            @tf.function
            def axioms(cite_a, cite_b, features_x, labels_z, p_forall, p_exists, p_forall_cites, p_exists_cites):
                features_x = ltn.Variable("x", features_x)
                labels_z = ltn.Variable("z", labels_z)
                cite_a = ltn.Variable("ca", cite_a)
                cite_b = ltn.Variable("cb", cite_b)

                axioms = []

                axioms.append(Forall(
                        ltn.diag(features_x,labels_z),
                        Exists(
                            (l1),
                            Document_type([features_x,l1]),
                            mask=equals([l1, labels_z]),
                            p=p_exists
                        ),
                        p=p_forall
                    )
                )

                axioms.append(Forall(
                                    ltn.diag(cite_a, cite_b),
                                    Exists(
                                        (l1),
                                        And(Document_type([cite_a,l1]),Document_type([cite_b,l1])),
                                        p=p_exists_cites
                                        ),
                                    p=p_forall_cites
                                    )
                            )
                
                return formula_aggregator(axioms).tensor

            # initialize all layers
            features_x, labels_z = next(train_set.as_numpy_iterator())
            axioms(cites_a, cites_b, features_x, labels_z, p_forall, p_exists, p_forall_cites, p_exists_cites)

            # training
            optimizer = tf.keras.optimizers.Adam(learning_rate)
            metrics_dict = {
                'train_loss': tf.keras.metrics.Mean(name="train_loss"),
                'train_accuracy': tf.keras.metrics.Mean(name="train_accuracy"),
                'test_loss': tf.keras.metrics.Mean(name="test_loss"),
                'test_accuracy': tf.keras.metrics.Mean(name="test_accuracy")    
            }

            @tf.function
            def train_step(cite_a, cite_b, features_x, labels_z, p_forall, p_exists, p_forall_cites, p_exists_cites):
                with tf.GradientTape() as tape:
                    loss = 1.- axioms(cite_a, cite_b, features_x, labels_z, p_forall, p_exists, p_forall_cites, p_exists_cites)
                gradients = tape.gradient(loss, logits_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, logits_model.trainable_variables))
            
            @tf.function
            def test_step(images_x, labels_z):
                predictions_x = tf.argmax(logits_model(images_x, training=False),axis=-1)

                match = tf.equal(predictions_x,tf.cast(labels_z,predictions_x.dtype))
                metrics_dict['test_accuracy'](tf.reduce_mean(tf.cast(match,tf.float32)))

            # training (with early stopping)
            total_training_time = 0
            best_accuracy = -1
            counter = 0
            nb_epochs_done = 0
            for epoch in range(nb_epochs):
                total_training_time += train_modified(cites_a, cites_b, train_set, train_step,
                                                    p_forall, p_exists, p_forall_cites, p_exists_cites, 1)
                val_accuracy = test_modified(val_set, test_step, metrics_dict)
                print("Val accuracy after epoch", epoch, ":", val_accuracy)
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    with open("best_model.pickle", "wb") as handle:
                        pickle.dump(logits_model.classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    counter = 0
                    nb_epochs_done = epoch + 1
                else:
                    if counter >= 2:
                        break
                    counter += 1

            # load best model
            if dataset == "CiteSeer":
                logits_model = Net_CiteSeer(0)
            elif dataset == "Cora":
                logits_model = Net_Cora(0)
            elif dataset == "PubMed":
                logits_model = Net_PubMed(0)
            with open("best_model.pickle", "rb") as handle:
                logits_model.classifier = pickle.load(handle)
            os.remove("best_model.pickle")

            # save trained model to a file
            with open(model_file_location, "wb") as handle:
                pickle.dump(logits_model.classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
            # testing
            start_time = time.time()
            accuracy = test_modified(test_set, test_step, metrics_dict)
            testing_time = time.time() - start_time
            
            # save results to a summary file
            information = {
                "algorithm": "LTN",
                "seed": seed,
                "nb_epochs": nb_epochs_done,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "p_forall": p_forall,
                "p_exists": p_exists,
                "p_forall_cites": p_forall_cites,
                "p_exists_cites": p_exists_cites,
                "dropout_rate": dropout_rate,
                "accuracy": float(accuracy),
                "training_time": float(total_training_time),
                "testing_time": float(testing_time),
                "model_file": model_file_name
            }
            with open(f'results/{dataset}/summary_final_{to_unsupervised}.json', "a") as outfile:
                json.dump(information, outfile)
                outfile.write('\n')

            # print results
            print("############################################")
            print("Seed: {} \nAccuracy: {} \nTraining time: {} \nTesting time: {}".format(seed, accuracy, 
                total_training_time, testing_time))
            print("############################################")
