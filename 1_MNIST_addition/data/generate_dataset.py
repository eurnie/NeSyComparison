import os
import random
import pathlib
import torchvision

def make_processed_dataset(raw_dataset, seed):
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

    return dataset

def write_to_file(dataset, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, "w+") as f:
        for entry in dataset:
            for elem in entry:
                f.write(str(elem))
                f.write(" ")
            f.write("\n")

def generate_dataset(seed):
    ROOT_FOLDER = pathlib.Path(__file__).parent

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))]
    )

    raw_train = torchvision.datasets.MNIST(
        root=str(ROOT_FOLDER), train=True, download=True, transform=transform)
    raw_test =  torchvision.datasets.MNIST(
        root=str(ROOT_FOLDER), train=False, download=True, transform=transform)

    processed_train = make_processed_dataset(raw_train, seed)
    processed_test = make_processed_dataset(raw_test, seed)

    print("The raw training set: " + str(len(raw_train)) + " entries")
    print("The raw testing set: " + str(len(raw_test)) + " entries")
    print("The processed training set: " + str(len(processed_train)) + " entries")
    print("The processed testing set: " + str(len(processed_test)) + " entries")

    write_to_file(processed_train, "../data/MNIST/processed/train.txt")
    write_to_file(processed_test, "../data/MNIST/processed/test.txt")