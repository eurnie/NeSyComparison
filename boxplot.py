import statistics
import matplotlib.pyplot as plt

#################
# dataset = "MNIST"
dataset = "FashionMNIST"
#################

assert dataset == "MNIST" or dataset == "FashionMNIST"
 
if dataset == "MNIST":
    data_deepproblog = []
    data_deepproblog_app = []
    data_deepstochlog = []
    data_ltn = []
    data_neurasp = []
    data_nn = []
    data_sl = []
elif dataset == "FashionMNIST":
    data_deepproblog = []
    data_deepproblog_app = []
    data_deepstochlog = []
    data_ltn = []
    data_neurasp = []
    data_nn = []
    data_sl = []
elif dataset == "CiteSeer":
    data_deepproblog = []
    data_deepproblog_app = []
    data_deepstochlog = []
    data_ltn = []
    data_neurasp = []
    data_nn = [73.60, 74.50, 74.50, 74.60, 73.90, 73.80, 63.50, 74.80, 74.20, 73.80]
    data_sl = [74.40, 77.20, 78.50, 76.70, 77.50, 77.20, 77.00, 77.00, 76.80, 74.80]
elif dataset == "Cora":
    data_deepproblog = []
    data_deepproblog_app = []
    data_deepstochlog = [83.40, 82.00, 83.10, 82.70, 84.00, 82.30, 84.40, 83.00, 82.80, 83.70]
    data_ltn = []
    data_neurasp = []
    data_nn = [31.90, 35.80, 35.00, 42.30, 35.30, 31.90, 31.80, 31.90, 32.90, 31.90]
    data_sl = [81.40, 82.80, 81.70, 83.90, 83.60, 81.90, 82.50, 82.20, 81.70, 83.20]

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