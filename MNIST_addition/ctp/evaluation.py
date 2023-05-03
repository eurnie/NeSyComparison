import numpy as np

def evaluate_on_mnist(test_path, predicate_to_idx, entity_to_idx, scoring_function):
    test_images_1 = []
    test_images_2 = []
    labels = []

    with open(test_path, "r") as f:
        for line in f.readlines():
            test_images_1.append(line[:-1].split(" ")[0])
            labels.append(line[:-1].split(" ")[1])
            test_images_2.append(line[:-1].split(" ")[2])

    predicates = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18']
    predicates_idx = [predicate_to_idx[x] for x in predicates]

    def predict(image_1, image_2):
        image_1_ids = [entity_to_idx[image_1]] * len(predicates_idx)
        image_2_ids = [entity_to_idx[image_2]] * len(predicates_idx)

        Xp = np.array(predicates_idx)
        Xs = np.array(image_1_ids)
        Xo = np.array(image_2_ids)

        return scoring_function(Xs, Xp, Xo)

    correct = 0
    total = 0
    for i in range(len(test_images_1)):
        total += 1
        
        image_1 = test_images_1[i]
        image_2 = test_images_2[i]
        y = int(labels[i])
        scores = np.array(predict(image_1, image_2))

        # print(f'actual: {y} -- predicted: {scores.argmax(0)}')
    
        if scores.argmax(0) == y:
            correct += 1

        if (i > 0) and (i % 10 == 0):
            print("Testing: {}/{} done.".format(i, len(test_images_1)))
            print("Current acc:", correct / total)
    return correct / len(test_images_1)