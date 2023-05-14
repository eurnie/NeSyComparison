import json

#################
dataset = "MNIST"
method = "nn"
#################

if (dataset == "MNIST"):
    param_tuning_file = f'MNIST_addition/{method}/results/MNIST/summary_param.json'

possible_parameter_list = ['dropout_rate', 'optimizer', 'learning_rate', 'batch_size', 'nb_epochs']
real_parameter_list = []

with open(param_tuning_file) as f:
    lines = f.readlines()

y = json.loads(lines[0])
for parameter in possible_parameter_list:
    if parameter in y:
        real_parameter_list.append(parameter)

data = []
accuracies = []
for parameter in real_parameter_list:
    data.append([])

for line in lines:
    y = json.loads(line)
    for i in range(len(real_parameter_list)):
        data[i].append(y[real_parameter_list[i]])
    accuracies.append(y['accuracy'] * 100)

max_value = max(accuracies)
indices = []

for i in range(len(accuracies)):
    if accuracies[i] == max_value:
        indices.append(i)

real_parameter_list = [str.replace("_", " ") for str in real_parameter_list]

with open('latex_table.txt', 'w+') as f:
    f.write('\\begin{table}\n')
    f.write('\\begin{adjustbox}{width=\\textwidth, pagecenter}\n')
    format = ['|c' if i == len(real_parameter_list) else 'c' for i in range(len(real_parameter_list) + 1)]
    format_string = ''.join(format)
    f.write(f'\\begin{{tabular}}{{{format_string}}}\n')
    f.write('\\toprule\n')
    f.write(f'{real_parameter_list[0]}')
    for i in range(1, len(real_parameter_list) + 1):
        if i >= len(real_parameter_list):
            f.write(f' & accuracy')
        else:
            f.write(f' & {real_parameter_list[i]}')
    f.write(' \\\ \n')
    f.write('\midrule\n')
    for i in range(len(data[0])):
        if i in indices:
            f.write('\\textbf{{{}}}'.format(data[0][i]))
            for j in range(1, len(data)):
                f.write(' & \\textbf{{{}}}'.format(data[j][i]))
            f.write(' & \\textbf{{{:,.2f}}}'.format(accuracies[i]))
        else:
            f.write('{}'.format(data[0][i]))
            for j in range(1, len(data)):
                f.write(f' & {data[j][i]}')
            f.write(' & {:,.2f}'.format(accuracies[i]))
        f.write(' \\\ \n')
    f.write('\\bottomrule\n')
    f.write('\end{tabular}\n')
    f.write('\end{adjustbox}\n')
    f.write(f'\caption{{Parameter tuning results for {method} on the {dataset} dataset.}}\n')
    f.write(f'\label{{tab:parameter_tuning_{method}_{dataset}}}\n')  
    f.write('\end{table}\n')