import json
from sklearn import linear_model

input_file = 'MNIST_addition/deepstochlog/results/mnist/kfold/summary_kfold.json'

with open(input_file) as f:
    lines = f.readlines()

parameters = []
all_parameter_values = []
accuracy = []
first = True

for line in lines:
    data = json.loads(line)

    if first:
        for parameter_name in data:
            if ((parameter_name != 'algorithm') and (parameter_name != 'seed') and 
                (parameter_name != 'size_val') and (parameter_name != 'accuracy') and 
                (parameter_name != 'model_file') and (parameter_name != 'avg_accuracy') and 
                (parameter_name != 'accuracies') and (parameter_name != 'model_files_dir') and 
                (parameter_name != 'method')):
                parameters.append(parameter_name)
        first = False
        print('This is a parameter tuning file of {}.'.format(data['algorithm']))
        print('{} has {} different parameters.'.format(data['algorithm'], len(parameters)))

    parameter_values = []
    for parameter_name in parameters:
        parameter_values.append(data[parameter_name])

    all_parameter_values.append(parameter_values)
    if 'kfold' in input_file:
        accuracy.append(data['avg_accuracy'])
    else:
        accuracy.append(data['accuracy'])

regr = linear_model.LinearRegression()
regr.fit(all_parameter_values, accuracy)

for i in range(len(regr.coef_)):
    print(parameters[i], ':', regr.coef_[i])

# make predictions 
# diabetes_y_pred = regr.predict(x_test)