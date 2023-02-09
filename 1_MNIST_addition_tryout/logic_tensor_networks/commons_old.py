import time
from collections import defaultdict

def train(
        epochs,
        metrics_dict, 
        ds_train, 
        ds_test, 
        train_step, 
        test_step,
        csv_path=None,
        scheduled_parameters=defaultdict(lambda : {})
    ):
    """
    Args:
        epochs: int, number of training epochs.
        metrics_dict: dict, {"metrics_label": tf.keras.metrics instance}.
        ds_train: iterable dataset, e.g. using tf.data.Dataset.
        ds_test: iterable dataset, e.g. using tf.data.Dataset.
        train_step: callable function. the arguments passed to the function
            are the itered elements of ds_train.
        test_step: callable function. the arguments passed to the function
            are the itered elements of ds_test.
        csv_path: (optional) path to create a csv file, to save the metrics.
        scheduled_parameters: (optional) a dictionary that returns kwargs for
            the train_step and test_step functions, for each epoch.
            Call using scheduled_parameters[epoch].
    """
    template = "Epoch {}"
    for metrics_label in metrics_dict.keys():
        template += ", %s: {:.4f}" % metrics_label
    if csv_path is not None:
        csv_file = open(csv_path,"w+")
        headers = ",".join(["Epoch"]+list(metrics_dict.keys()))
        csv_template = ",".join(["{}" for _ in range(len(metrics_dict)+1)])
        csv_file.write(headers+"\n")
    
    for epoch in range(epochs):
        for metrics in metrics_dict.values():
            metrics.reset_states()

        for batch_elements in ds_train:
            train_step(*batch_elements,**scheduled_parameters[epoch])
        for batch_elements in ds_test:
            test_step(*batch_elements,**scheduled_parameters[epoch])

        metrics_results = [metrics.result() for metrics in metrics_dict.values()]
        print(template.format(epoch,*metrics_results))
        if csv_path is not None:
            csv_file.write(csv_template.format(epoch,*metrics_results)+"\n")
            csv_file.flush()
    if csv_path is not None:
        csv_file.close()

def train_modified(ds_train, train_step, ds_test, test_step, metrics_dict, scheduled_parameters, 
    nb_epochs, batch_size, log_iter, SEED_PYTHON, SEED_NUMPY, SEED_TORCH):
    total_training_time = 0
    highest_accuracy = 0
    highest_accuracy_index = 0

    for epoch in range(0, nb_epochs):
        nb_examples_seen = 0
        
        for batch_elements in ds_train:
            # training
            start_time = time.time()
            train_step(*batch_elements,**scheduled_parameters[epoch])
            total_training_time += time.time() - start_time
            nb_examples_seen += batch_size

            # evaluation
            if (nb_examples_seen % log_iter) == 0:
                accuracy = test_modified(ds_test, test_step, metrics_dict, scheduled_parameters, epoch)

                if (accuracy > highest_accuracy):
                    highest_accuracy = accuracy
                    highest_accuracy_index = (epoch * 30000) + nb_examples_seen

                log_file = "results/results_ltn_{}_{}_{}_{}_{}.txt".format(SEED_PYTHON, SEED_NUMPY, SEED_TORCH, batch_size, nb_epochs)

                with open(log_file, "a") as f:
                    f.write(str((epoch * 30000) + nb_examples_seen))
                    f.write(" ")
                    f.write(str(total_training_time))
                    f.write(" ")
                    f.write(str(accuracy))
                    f.write(" ")
                    f.write("\n")

                print("############################################")
                print("Number of entries: ", (epoch * 30000) + nb_examples_seen)
                print("Total training time: ", total_training_time)
                print("Accuracy: ", accuracy)
                print("############################################")

    print("The highest accuracy was {} and was reached (the first time) after seeing {} samples.".format(highest_accuracy, highest_accuracy_index))

def test_modified(ds_test, test_step, metrics_dict, scheduled_parameters, epoch):
    template = "Epoch {}"
    for metrics_label in metrics_dict.keys():
        template += ", %s: {:.4f}" % metrics_label
    for metrics in metrics_dict.values():
        metrics.reset_states()
    for batch_elements in ds_test:
        test_step(*batch_elements,**scheduled_parameters[epoch])
    metrics_results = [metrics.result() for metrics in metrics_dict.values()]
    return metrics_results[3].numpy()