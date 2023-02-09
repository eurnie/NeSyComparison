import sys
sys.path.append('../')
import torch
from torch import nn
from deepproblog.dataset import DataLoader, Dataset
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.heuristics import geometric_mean
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.utils import get_configuration, format_time_precise, config_to_string
from data.import_data import MNIST_train, MNIST_test
from deepproblog.query import Query
from problog.logic import Term, list2term, Constant

sys.path.append('../')
from data.import_data import GeneralMNISTOperator, datasets

def addition_deepproblog(n: int, dataset: str, seed=None):
    """Returns a dataset for binary addition"""
    return MNISTOperator(
        dataset_name=dataset,
        function_name="addition" if n == 1 else "multi_addition",
        operator=sum,
        size=n,
        arity=2,
        seed=seed,
    )

class MNIST(Dataset):
    def __len__(self):
        return len(self.data)

    def to_query(self, i):
        l = Constant(self.data[i][1])
        return Query(
            Term("digit", Term("tensor", Term(self.dataset, Term("a"))), l),
            substitution={Term("a"): Constant(i)},
        )

    def __init__(self, dataset):
        self.dataset = dataset
        self.data = datasets[dataset]

class MNISTOperator(GeneralMNISTOperator, Dataset):
    def to_query(self, i: int) -> Query:
        """Generate queries"""
        mnist_indices = self.data[i]
        expected_result = self._get_label(i)

        # Build substitution dictionary for the arguments
        subs = dict()
        var_names = []
        for i in range(self.arity):
            inner_vars = []
            for j in range(self.size):
                t = Term(f"p{i}_{j}")
                subs[t] = Term(
                    "tensor",
                    Term(
                        self.dataset_name,
                        Constant(mnist_indices[i][j]),
                    ),
                )
                inner_vars.append(t)
            var_names.append(inner_vars)

        # Build query
        if self.size == 1:
            return Query(
                Term(
                    self.function_name,
                    *(e[0] for e in var_names),
                    Constant(expected_result),
                ),
                subs,
            )
        else:
            return Query(
                Term(
                    self.function_name,
                    *(list2term(e) for e in var_names),
                    Constant(expected_result),
                ),
                subs,
            )

parameters = {
    "method": ["exact"],            # "gm" or "exact"
    "N": [1],                       # 1, 2 or 3
    "pretrain": [None],             # number of pretrained model or None
    "exploration": [False],         # True or False
    "run": range(5),                # range(5)
}

class addition_neural_net(nn.Module):
    def __init__(self, N=10, with_softmax=True, size=16 * 4 * 4):
        super(addition_neural_net, self).__init__()
        self.with_softmax = with_softmax
        self.size = size
        if with_softmax:
            if N == 1:
                self.softmax = nn.Sigmoid()
            else:
                self.softmax = nn.Softmax(1)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, N),
        )

    def forward(self, x):
        # x = x.unsqueeze(0)
        x = self.encoder(x)
        x = x.view(-1, self.size)
        x = self.classifier(x)
        if self.with_softmax:
            x = self.softmax(x)
        return x

configuration = get_configuration(parameters, 0)
torch.manual_seed(configuration["run"])

name = "addition_" + config_to_string(configuration) + "_" + format_time_precise()
print(name)

train_set = addition_deepproblog(configuration["N"], "train")
test_set = addition_deepproblog(configuration["N"], "test")
network = addition_neural_net()

pretrain = configuration["pretrain"]
if pretrain is not None and pretrain > 0:
    network.load_state_dict(
        torch.load("models/pretrained/all_{}.pth".format(configuration["pretrain"]))
    )
net = Network(network, "mnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

model = Model("addition.pl", [net])
if configuration["method"] == "exact":
    if configuration["exploration"] or configuration["N"] > 2:
        print("Not supported?")
        exit()
    model.set_engine(ExactEngine(model), cache=True)
elif configuration["method"] == "gm":
    model.set_engine(
        ApproximateEngine(
            model, 1, geometric_mean, exploration=configuration["exploration"]
        )
    )
model.add_tensor_source("train", MNIST_train)
model.add_tensor_source("test", MNIST_test)

loader_train = DataLoader(train_set, batch_size=4, shuffle=False)

epochs = 10
for t in range(epochs):
    print("Epoch {}\n-------------------------------".format(t+1))

    # train
    train = train_model(model, loader_train, 1, log_iter=100, profile=0)
    model = train.model

    # test
    model.eval()
    correct_predictions = 0

    for i, query in enumerate(test_set.to_queries()):
        test_query = query.variable_output()
        answer = model.solve([test_query])[0]
        actual = str(query.output_values()[0])

        if len(answer.result) == 0:
            predicted = None
        else:
            max_ans = max(answer.result, key=lambda x: answer.result[x])
            predicted = str(max_ans.args[query.output_ind[0]])
        if (predicted == actual):
            correct_predictions += 1

    print("ACCURACY: ",correct_predictions/len(test_set))
    
    for nb_entries in range(0, len(train_set)+1-100, 100):
    loader = DataLoader(train_set[nb_entries:nb_entries+100], batch_size, False)
    
    start_time = time.time()
    train = train_model(model, loader, 1, log_iter=1000, profile=0)
    total_training_time += time.time() - start_time

    model.eval()
    correct_predictions = 0

    for i, query in enumerate(test_set.to_queries()):
        test_query = query.variable_output()
        answer = model.solve([test_query])[0]
        actual = str(query.output_values()[0])

        if len(answer.result) == 0:
            predicted = None
        else:
            max_ans = max(answer.result, key=lambda x: answer.result[x])
            predicted = str(max_ans.args[query.output_ind[0]])
        if (predicted == actual):
            correct_predictions += 1

    accuracy = correct_predictions/len(test_set)

    with open("results.txt", "w+") as f:
        f.write(nb_entries)
        f.write(" ")
        f.write(total_training_time)
        f.write(" ")
        f.write(accuracy)
        f.write(" ")
        f.write("\n")

    print("###################################")
    print("Number of entries: ", nb_entries)
    print("Total training time: ", total_training_time)
    print("Accuracy: ", accuracy)
    print("###################################")
