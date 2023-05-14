import statistics
import matplotlib.pyplot as plt

#################
# dataset = "MNIST"
dataset = "FashionMNIST"
#################

assert dataset == "MNIST" or dataset == "FashionMNIST"
 
if dataset == "MNIST":
    data_deepproblog = [97.90, 97.66, 97.94, 97.64, 97.62, 97.84, 97.76, 97.82, 97.80, 97.80]
    data_deepproblog_app = [97.54, 97.52, 80.80, 80.08, 62.46, 62.84, 64.16, 62.46, 63.24, 65.44]
    data_deepstochlog = [97.52, 97.92, 97.68, 97.82, 97.76, 97.70, 97.72, 97.46, 97.90, 97.30]
    data_ltn = [96.84, 97.14, 96.24, 79.40, 62.30, 79.10, 97.64, 63.88, 97.52, 64.72]
    data_neurasp = [97.68, 96.98, 97.98, 97.14, 98.04, 97.58, 97.52, 98.02, 97.60, 97.96]
    data_nn = [59.72, 65.14, 60.70, 55.06, 71.32, 75.22, 72.70, 74.60, 67.20, 61.60]
    data_sl = [97.26, 97.06, 97.30, 96.56, 97.64, 97.62, 97.12, 97.04, 97.40, 97.28]
elif dataset == "FashionMNIST":
    data_deepproblog = [79.16, 77.38, 78.96, 78.62, 79.36, 79.32, 78.68, 78.94, 78.80, 79.24]
    data_deepproblog_app = [40.24, 52.44, 38.54, 53.46, 52.00, 38.46, 40.72, 63.00, 42.70, 52.52]
    data_deepstochlog = [77.36, 77.98, 80.40, 77.78, 77.10, 80.16, 78.08, 78.78, 78.04, 79.70]
    data_ltn = [60.10, 69.02, 75.52, 71.60, 64.66, 72.52, 56.78, 53.76, 44.84, 65.38]
    data_neurasp = [77.66, 79.30, 79.00, 79.16, 78.54, 78.70, 78.08, 78.84, 78.72, 78.62]
    data_nn = [46.18, 46.38, 32.48, 40.94, 42.40, 40.42, 43.98, 47.10, 39.64, 41.74]
    data_sl = [74.24, 76.26, 76.88, 73.32, 77.70, 74.88, 75.80, 74.42, 75.48, 75.30]
elif dataset == "CiteSeer":
    data_deepproblog = []
    data_deepproblog_app = []
    data_deepstochlog = [78.00, 77.30, 77.40, 78.60, 78.60, 79.50, 77.60, 77.40, 78.80, 78.40]
    data_ltn = [70.19, 65.99, 63.13, 69.79, 76.97, 74.83, 75.15, 67.68, 76.51, 77.26]
    data_neurasp = [73.70, 72.80, 74.60, 74.50, 72.80, 73.20, 72.00, 73.60, 72.40, 73.20]
    data_nn = [73.50, 72.70, 74.20, 74.20, 72.20, 73.20, 72.40, 71.80, 74.00, 73.30]
    data_sl = [73.80, 75.50, 75.40, 76.80, 74.70, 75.50, 76.90, 75.80, 76.90, 74.00]

data = [data_deepproblog, data_deepproblog_app, data_deepstochlog, data_ltn, data_neurasp, data_nn, data_sl]
methods = ['DeepProbLog', 'DeepProbLog (approximate)', 'DeepStochLog', 'Logic Tensor Networks', 'NeurASP', 'NN baseline', 'Semantic Loss']

# create latex table
with open('latex_table.txt', 'w+') as f:
    f.write('\\begin{table}\n')
    f.write('\\begin{adjustbox}{width=1.3\\textwidth, pagecenter}\n')
    f.write('\\begin{tabular}{lcccccccccccc}\n')
    f.write('\\toprule\n')
    f.write(f'\multicolumn{{1}}{{c}}{{}}')
    for i in range(12):
        if i == 10:
            f.write(f' & \\textbf{{AVG}}')
        elif i == 11:
            f.write(f' & \\textbf{{STD}}')
        else:
            f.write(f' & {i}')
    f.write(' \\\ \n')
    f.write('\midrule\n')
    for i in range(len(methods)):
        accuracies = data[i]
        assert len(accuracies) == 10
        method = methods[i]
        f.write(method)
        for acc in accuracies:
             f.write(' & {:,.2f}'.format(acc))
        avg = sum(accuracies) / len(accuracies)
        f.write(' & \\textbf{{{:,.2f}}}'.format(avg))
        std = statistics.stdev(accuracies)
        f.write(' & \\textbf{{{:,.2f}}}'.format(std))
        f.write(' \\\ \n')
    f.write('\\bottomrule\n')
    f.write('\end{tabular}\n')
    f.write('\end{adjustbox}\n')
    f.write(f'\caption{{Accuracy on the test set for the {dataset} addition problem. The columns indicate the seed that was used for shuffling the training dataset and initializing the neural networks. The final two columns show the average accuracy and the standard deviation for each method.}}\n')
    f.write(f'\label{{tab:{dataset}_final}}\n')  
    f.write('\end{table}\n')

# create boxplot
# fig = plt.figure(figsize =(10, 7))
# plt.rcParams['font.size'] = '20'
# ax = fig.add_subplot(111)

# bp = ax.boxplot(data, vert = 0)

# ax.set_yticklabels(['NN\nbaseline', 'DeepProbLog', 'DeepProblog\n(approximate)', 'DeepStochLog', 'Logic\nTensor\nNetworks', 'NeurASP', 'Semantic\nLoss'])
# # plt.title("MNIST addition")
# plt.xlabel("Accuracy test set (%)")
# plt.show()