import matplotlib.pyplot as plt
 
data_deepproblog = [97.90, 97.66, 97.94, 97.64, 97.62, 97.84, 97.76, 97.82, 97.80, 97.80]
data_deepproblog_app = [97.54, 97.52, 80.80, 80.08, 62.46, 62.84, 64.16, 62.46, 63.24, 65.44]
data_deepstochlog = [97.52, 97.92, 97.68, 97.82, 97.76, 97.70, 97.72, 97.46, 97.90, 97.30]
data_ltn = [96.84, 97.14, 96.24, 79.40, 62.30, 79.10, 97.64, 63.88, 97.52, 64.72]
data_neurasp = [97.68, 96.98, 97.98, 97.14, 98.04, 97.58, 97.52, 98.02, 97.60, 97.96]
data_nn = [59.72, 65.14, 60.70, 55.06, 71.32, 75.22, 72.70, 74.60, 67.20, 61.60]
data_sl = [97.26, 97.06, 97.30, 96.56, 97.64, 97.62, 97.12, 97.04, 97.40, 97.28]

data = [data_nn, data_deepproblog, data_deepproblog_app, data_deepstochlog, data_ltn, data_neurasp, data_sl]

fig = plt.figure(figsize =(10, 7))
ax = fig.add_subplot(111)
 
bp = ax.boxplot(data, vert = 0)

ax.set_yticklabels(['NN baseline', 'DeepProbLog', 'DeepProblog (approximate)', 'DeepStochLog', 'Logic Tensor Networks', 'NeurASP', 'Semantic Loss'])
# plt.title("MNIST addition")
plt.xlabel("Accuracy test set (%)")
plt.show()