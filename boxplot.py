import matplotlib.pyplot as plt

#################
dataset = "Cora"
#################

assert dataset == "MNIST" or dataset == "FashionMNIST" or dataset == "CiteSeer" or dataset == "Cora"
 
if dataset == "MNIST":
	data_deepproblog = [97.90, 97.66, 97.94, 97.64, 97.62, 97.84, 97.76, 97.82, 97.80, 97.80]
	data_deepproblog_app = [98.06, 65.66, 65.74, 65.20, 98.02, 63.24, 97.96, 97.90, 51.06, 98.18]
	data_deepstochlog = [97.72, 97.66, 98.00, 97.48, 97.88, 97.80, 97.78, 97.68, 97.80, 97.46]
	data_ltn = [96.24, 79.66, 97.20, 80.26, 96.86, 95.56, 97.76, 97.30, 80.34, 79.96]
	data_neurasp = [98.34, 97.02, 97.74, 98.08, 98.02, 98.16, 98.18, 98.00, 97.80, 98.62]
	data_nn = [75.66, 64.64, 65.66, 54.48, 74.60, 66.82, 63.58, 58.52, 62.26, 50.86]
	data_sl = [97.68, 97.62, 97.62, 97.16, 97.74, 96.38, 97.64, 97.50, 97.48, 96.88]
elif dataset == "FashionMNIST":
    data_deepproblog = [79.58, 77.58, 78.24, 78.72, 78.78, 78.26, 78.06, 78.64, 78.76, 79.38]
    data_deepproblog_app = [52.90, 54.60, 51.00, 56.53, 67.10, 42.00, 56.70, 45.27, 56.53,  54.03]
    data_deepstochlog = [77.84, 79.10, 79.40, 78.72, 79.68, 79.32, 78.08, 79.80, 80.70, 79.16]
    data_ltn = [76.02, 68.44, 76.60, 72.42, 51.68, 73.72, 40.08, 51.84, 52.60, 77.90]
    data_neurasp = [77.42, 80.30, 79.08, 78.78, 79.04, 79.34, 79.54, 80.50, 80.30, 78.78]
    data_nn = [49.94, 47.80, 38.48, 46.10, 44.44, 43.90, 41.58, 36.02, 41.64, 52.30]
    data_sl = [73.84, 74.38, 75.74, 77.44, 73.64, 76.04, 71.16, 75.30, 76.28, 73.92]
elif dataset == "CiteSeer":
    data_deepproblog = [76.80, 77.10, 76.50, 77.30, 76.50, 77.30, 76.80, 75.00, 77.74, 77.10]
    data_deepproblog_app = [72.00, 73.40, 70.90, 71.90, 72.50, 72.60, 72.70, 72.60, 71.40, 71.10]
    data_deepstochlog = [77.30, 77.50, 77.30, 78.60, 78.60, 79.50, 78.20, 77.80, 78.90, 78.90]
    data_ltn = [76.40, 75.12, 74.81, 69.63, 76.01, 75.96, 76.59, 66.62, 72.76, 76.04]
    data_neurasp = []
    data_nn = [73.60, 74.50, 74.50, 74.60, 73.90, 73.80, 63.50, 74.80, 74.20, 73.80]
    data_sl = [74.40, 77.20, 78.50, 76.70, 77.50, 77.20, 77.00, 77.00, 76.80, 74.80]
elif dataset == "Cora":
    data_deepproblog = [83.90, 84.60, 84.40, 83.90, 83.70, 83.90, 83.30, 84.40, 84.60, 83.80]
    data_deepproblog_app = [79.80, 77.80, 78.00, 78.70, 78.10, 78.80, 79.70, 77.90, 77.70, 79.10]
    data_deepstochlog = [84.30, 82.10, 83.40, 83.70, 84.40, 83.80, 83.60, 84.00, 84.60, 84.20]
    data_ltn = [32.59, 40.72, 38.06, 32.27, 33.38, 32.15, 32.02, 32.51, 54.71, 42.12]
    data_neurasp = []
    data_nn = [31.90, 35.80, 35.00, 42.30, 35.30, 31.90, 31.80, 31.90, 32.90, 31.90]
    data_sl = [81.40, 82.80, 81.70, 83.90, 83.60, 81.90, 82.50, 82.20, 81.70, 83.20]

data = [data_neurasp, data_deepstochlog, data_deepproblog_app, data_deepproblog, data_ltn, data_sl, data_nn]
methods = ['NeurASP', 'DeepStochLog', 'DeepProbLog\n(approximate)', 'DeepProbLog', 'Logic\nTensor\nNetworks', 'Semantic\nLoss', 'NN\nbaseline']

# # create latex table
# with open('latex_table.txt', 'w+') as f:
#     f.write('\\begin{table}\n')
#     f.write('\\begin{adjustbox}{width=1.3\\textwidth, pagecenter}\n')
#     f.write('\\begin{tabular}{lcccccccccccc}\n')
#     f.write('\\toprule\n')
#     f.write(f'\multicolumn{{1}}{{c}}{{}}')
#     for i in range(12):
#         if i == 10:
#             f.write(f' & \\textbf{{AVG}}')
#         elif i == 11:
#             f.write(f' & \\textbf{{STD}}')
#         else:
#             f.write(f' & {i}')
#     f.write(' \\\ \n')
#     f.write('\midrule\n')
#     for i in range(len(methods)):
#         accuracies = data[i]
#         assert len(accuracies) == 10
#         method = methods[i]
#         f.write(method)
#         for acc in accuracies:
#              f.write(' & {:,.2f}'.format(acc))
#         avg = sum(accuracies) / len(accuracies)
#         f.write(' & \\textbf{{{:,.2f}}}'.format(avg))
#         std = statistics.stdev(accuracies)
#         f.write(' & \\textbf{{{:,.2f}}}'.format(std))
#         f.write(' \\\ \n')
#     f.write('\\bottomrule\n')
#     f.write('\end{tabular}\n')
#     f.write('\end{adjustbox}\n')
#     f.write(f'\caption{{Accuracy on the test set for the {dataset} addition problem. The columns indicate the seed that was used for shuffling the training dataset and initializing the neural networks. The final two columns show the average accuracy and the standard deviation for each method.}}\n')
#     f.write(f'\label{{tab:{dataset}_final}}\n')  
#     f.write('\end{table}\n')

# create boxplot
fig = plt.figure(figsize =(10, 7))
plt.rcParams['font.size'] = '20'
ax = fig.add_subplot(111)

bp = ax.boxplot(data, vert = 0)

ax.set_yticklabels(methods)
plt.title("Cora")
plt.xlabel("Accuracy test set (%)")
plt.show()
