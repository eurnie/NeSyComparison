import matplotlib.pyplot as plt
 
data_deepproblog = []
data_deepproblog_app = []
data_deepstochlog = [78.00, 77.30, 77.40, 78.60, 78.60, 79.50, 77.60, 77.40, 78.80, 78.40]
data_ltn = [70.19, 65.99, 63.13, 69.79, 76.97, 74.83, 75.15, 67.68, 76.51, 77.26]
data_neurasp = [73.70, 72.80, 74.60, 74.50, 72.80, 73.20, 72.00, 73.60, 72.40, 73.20]
data_nn = [73.50, 72.70, 74.20, 74.20, 72.20, 73.20, 72.40, 71.80, 74.00, 73.30]
data_sl = [73.80, 75.50, 75.40, 76.80, 74.70, 75.50, 76.90, 75.80, 76.90, 74.00]

data = [data_nn, data_deepstochlog, data_ltn, data_neurasp, data_sl]

fig = plt.figure(figsize =(10, 7))
plt.rcParams['font.size'] = '20'
ax = fig.add_subplot(111)
 
bp = ax.boxplot(data, vert = 0)

ax.set_yticklabels(['NN\nbaseline', 'DeepStochLog', 'Logic\nTensor\nNetworks', 'NeurASP', 'Semantic\nLoss'])
# plt.title("CiteSeer")
plt.xlabel("Accuracy test set (%)")
plt.show()