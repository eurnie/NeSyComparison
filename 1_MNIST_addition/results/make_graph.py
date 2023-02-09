import glob
import matplotlib.pyplot as plt
import numpy as np

file_deepproblog = glob.glob("results_deepproblog*")[0]
file_deepstochlog = glob.glob("results_deepstochlog*")[0]
file_neurasp = glob.glob("results_neurasp*")[0]
file_ltn = glob.glob("results_ltn*")[0]

### DeepProbLog ###
with open(file_deepproblog) as f:
    lines = f.readlines()

x_deepproblog_examples = []
x_deepproblog_time = []
y_deepproblog = []

for line in lines:
    nb_entries = line.split(" ")[0]
    total_training_time = line.split(" ")[1]
    accuracy = line.split(" ")[2]

    x_deepproblog_examples.append(int(nb_entries))
    x_deepproblog_time.append(float(total_training_time))
    y_deepproblog.append(float(accuracy))

### DeepStochLog ###
with open(file_deepstochlog) as f:
    lines = f.readlines()

x_deepstochlog_examples = []
x_deepstochlog_time = []
y_deepstochlog = []

for line in lines:
    nb_entries = line.split(" ")[0]
    total_training_time = line.split(" ")[1]
    accuracy = line.split(" ")[2]

    x_deepstochlog_examples.append(int(nb_entries))
    x_deepstochlog_time.append(float(total_training_time))
    y_deepstochlog.append(float(accuracy))

### NeurASP ###
with open(file_neurasp) as f:
    lines = f.readlines()

x_neurasp_examples = []
x_neurasp_time = []
y_neurasp = []

for line in lines:
    nb_entries = line.split(" ")[0]
    total_training_time = line.split(" ")[1]
    accuracy = line.split(" ")[2]

    x_neurasp_examples.append(int(nb_entries))
    x_neurasp_time.append(float(total_training_time))
    y_neurasp.append(float(accuracy))

### Logic Tensor Networks ###
with open(file_ltn) as f:
    lines = f.readlines()

x_ltn_examples = []
x_ltn_time = []
y_ltn = []

for line in lines:
    nb_entries = line.split(" ")[0]
    total_training_time = line.split(" ")[1]
    accuracy = line.split(" ")[2]

    x_ltn_examples.append(int(nb_entries))
    x_ltn_time.append(float(total_training_time))
    y_ltn.append(float(accuracy))

### FINALIZE EXAMPLES GRAPH ###
plt.plot(x_deepproblog_examples, y_deepproblog, label='DeepProbLog')   
plt.plot(x_deepstochlog_examples, y_deepstochlog, label='DeepStochLog')   
plt.plot(x_neurasp_examples, y_neurasp, label='NeurASP')
plt.plot(x_ltn_examples, y_ltn, label='LTN')   
plt.xlabel("Number of examples")
plt.ylabel("Accuracy on test set")
plt.title("MNIST addition")
plt.yticks(np.arange(0, 1.1, 0.1))
plt.ylim([0.8, 1])
plt.legend(loc="lower right")
plt.show()

### FINALIZE TRAINING TIME GRAPH ###
plt.plot(x_deepproblog_time, y_deepproblog, label='DeepProbLog')
plt.plot(x_deepstochlog_time, y_deepstochlog, label='DeepStochLog')   
plt.plot(x_neurasp_time, y_neurasp, label='NeurASP')
plt.plot(x_ltn_time, y_ltn, label='LTN')   
plt.xlabel("Training time (seconds)")
plt.ylabel("Accuracy on test set")
plt.title("MNIST addition")
plt.yticks(np.arange(0, 1.1, 0.1))
plt.ylim([0.8, 1])
plt.legend(loc="lower right")
plt.show()