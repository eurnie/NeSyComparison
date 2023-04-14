import os
import random
import pathlib
import torchvision

ROOT_FOLDER = pathlib.Path(__file__).parent

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))]
)

# import MNIST dataset
raw_train_mnist = torchvision.datasets.MNIST(
    root=str(ROOT_FOLDER), train=True, download=True, transform=transform)
raw_test_mnist = torchvision.datasets.MNIST(
    root=str(ROOT_FOLDER), train=False, download=True, transform=transform)

# import MNIST fashion dataset
raw_train_fashion_mnist = torchvision.datasets.FashionMNIST(
    root=str(ROOT_FOLDER), train=True, download=True, transform=transform)
raw_test_fashion_mnist = torchvision.datasets.FashionMNIST(
    root=str(ROOT_FOLDER), train=False, download=True, transform=transform)

# process the raw dataset and shuffle
def make_processed_dataset(raw_dataset, seed, label_noise):
    indices = list(range(len(raw_dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    dataset = []

    for i in range(0, len(indices)-1, 2):
        new_entry = []

        index_digit_1 = indices[i]
        index_digit_2 = indices[i+1]
        sum = raw_dataset[indices[i]][1] + raw_dataset[indices[i+1]][1]
        
        new_entry.append(index_digit_1)
        new_entry.append(index_digit_2)
        new_entry.append(sum)

        dataset.append(new_entry)

    examples = random.sample(dataset, round(label_noise * len(dataset)))
    for i in range(len(examples)):
        examples[i][2] = random.randint(0,18)

    return dataset

# write the given dataset to the file with the given filename
def write_to_file(dataset, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, "w+") as f:
        for entry in dataset:
            for elem in entry:
                f.write(str(elem))
                f.write(" ")
            f.write("\n")

# generate the mnist addition dataset
def generate_dataset_mnist(seed, label_noise):
    processed_train = make_processed_dataset(raw_train_mnist, seed, label_noise)
    processed_test = make_processed_dataset(raw_test_mnist, seed, 0)

    print("The raw training set: " + str(len(raw_train_mnist)) + " entries")
    print("The raw testing set: " + str(len(raw_test_mnist)) + " entries")
    print("The processed training set: " + str(len(processed_train)) + " entries")
    print("The processed testing set: " + str(len(processed_test)) + " entries")

    write_to_file(processed_train, "../data/MNIST/processed/train.txt")
    write_to_file(processed_test, "../data/MNIST/processed/test.txt")

# generate the fashion mnist addition dataset
def generate_dataset_fashion_mnist(seed, label_noise):
    processed_train = make_processed_dataset(raw_train_fashion_mnist, seed, label_noise)
    processed_test = make_processed_dataset(raw_test_fashion_mnist, seed, 0)

    print("The raw training set: " + str(len(raw_train_fashion_mnist)) + " entries")
    print("The raw testing set: " + str(len(raw_test_fashion_mnist)) + " entries")
    print("The processed training set: " + str(len(processed_train)) + " entries")
    print("The processed testing set: " + str(len(processed_test)) + " entries")

    write_to_file(processed_train, "../data/FashionMNIST/processed/train.txt")
    write_to_file(processed_test, "../data/FashionMNIST/processed/test.txt")