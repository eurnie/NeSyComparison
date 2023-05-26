import sys

sys.path.append('..')
from data.generate_dataset import generate_dataset_mnist, generate_dataset_fashion_mnist

def write_to_file(dataset, filename):   
    with open(filename, 'w+') as f:
        for d1, d2, sum in dataset:
            string_to_write = ''
            if sum == 0:
                string_to_write += 'zero('
            elif sum == 1:
                string_to_write += 'one('
            elif sum == 2:
                string_to_write += 'two('
            elif sum == 3:
                string_to_write += 'three('
            elif sum == 4:
                string_to_write += 'four('
            elif sum == 5:
                string_to_write += 'five('
            elif sum == 6:
                string_to_write += 'six('
            elif sum == 7:
                string_to_write += 'seven('
            elif sum == 8:
                string_to_write += 'eight('
            elif sum == 9:
                string_to_write += 'nine('
            elif sum == 10:
                string_to_write += 'ten('
            elif sum == 11:
                string_to_write += 'eleven('
            elif sum == 12:
                string_to_write += 'twelve('
            elif sum == 13:
                string_to_write += 'thirteen('
            elif sum == 14:
                string_to_write += 'fourteen('
            elif sum == 15:
                string_to_write += 'fifteen('
            elif sum == 16:
                string_to_write += 'sixteen('
            elif sum == 17:
                string_to_write += 'seventeen('
            elif sum == 18:
                string_to_write += 'eighteen('
            elif sum == 'X':
                string_to_write += 'unknown('
            string_to_write += str(d1)
            string_to_write += ','
            string_to_write += str(d2)
            string_to_write += ').'
            f.write(string_to_write)
            f.write('\n')

def write_to_file_txt(dataset, filename):   
    with open(filename, 'w+') as f:
        for d1, d2, sum in dataset:
            string_to_write = ''
            string_to_write += str(d1)
            string_to_write += ' '
            string_to_write += str(d2)
            string_to_write += ' '
            string_to_write += str(sum)
            f.write(string_to_write)
            f.write('\n')

def create_datasets(train_path, test_path, size_val):
    percentage_of_original_train_dataset = 0.001
    percentage_of_original_val_dataset = 0.001
    percentage_of_original_test_dataset = 0.001
    split_index = round(size_val * 30000)

    for dataset_name in ["train", "val", "test"]:
        
        if dataset_name == "train":
            with open(train_path) as f:
                entries = f.readlines()
            start = split_index
            end = split_index + round(percentage_of_original_train_dataset * ((1-size_val)*30000))
            write_dataset_name = "train"
        elif dataset_name == "val":
            with open(train_path) as f:
                entries = f.readlines()
            start = 0
            end = round(percentage_of_original_val_dataset * split_index)
            write_dataset_name = "train"
        elif dataset_name == "test":
            with open(test_path) as f:
                entries = f.readlines()
            start = 0
            end = round(percentage_of_original_test_dataset * 5000)
            write_dataset_name = "test"

        dataset = []
        dataset_unknowns = []

        for i in range(start, end):
            index_digit_1 = write_dataset_name + "-" + entries[i].split(" ")[0]
            index_digit_2 = write_dataset_name + "-" + entries[i].split(" ")[1]
            sum = int(entries[i].split(" ")[2])
            dataset.append((index_digit_1, index_digit_2, sum))

        if dataset_name != "train":
            for i in range(start, end):
                index_digit_1 = write_dataset_name + "-" + entries[i].split(" ")[0]
                index_digit_2 = write_dataset_name + "-" + entries[i].split(" ")[1]
                dataset_unknowns.append((index_digit_1, index_digit_2, 'X'))

        if dataset_name == 'train':
            write_to_file(dataset, "{}.nl".format(dataset_name))
        else:
            write_to_file(dataset_unknowns, "{}.nl".format(dataset_name))
            write_to_file_txt(dataset, "{}.txt".format(dataset_name))

    filenames = ['rules.nl', 'train.nl', 'val.nl', 'test.nl']
    with open('mnist_addition.nl', 'w+') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
                outfile.write('\n')

def generate_dataset(dataset, label_noise, size_val, seed):
    if dataset == "MNIST":
        generate_dataset_mnist(seed, label_noise)
        create_datasets("../data/MNIST/processed/train.txt", "../data/MNIST/processed/test.txt", size_val)
    elif dataset == "FashionMNIST":
        generate_dataset_fashion_mnist(seed, label_noise)
        create_datasets("../data/FashionMNIST/processed/train.txt", "../data/FashionMNIST/processed/test.txt", size_val)
