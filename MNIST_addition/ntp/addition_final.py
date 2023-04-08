import os
import sys
import time
import json
import torch
import random
import logging
import multiprocessing
import numpy as np
from torch import nn, optim, Tensor
from ctp.training.data import Data_MNIST
from ctp.training.batcher import Batcher
from ctp.kernels import BaseKernel, GaussianKernel
from ctp.smart.kb import NeuralKB
from ctp.smart.simple import SimpleHoppy
from ctp.reformulators import BaseReformulator
from ctp.reformulators import StaticReformulator
from ctp.reformulators import LinearReformulator
from ctp.reformulators import AttentiveReformulator
from ctp.reformulators import MemoryReformulator
from ctp.reformulators import NTPReformulator
from ctp.regularizers import N2, N3
from typing import Tuple, Dict, Optional
from evaluation import evaluate_on_mnist
from custom_entity_embedding import CustomEntityEmbeddings

sys.path.append("..")
from data.generate_dataset import generate_dataset

logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=48, precision=5, suppress=True)
torch.set_num_threads(multiprocessing.cpu_count())
torch.autograd.set_detect_anomaly(True)

def metrics_to_str(metrics):
    return f'MRR {metrics["MRR"]:.6f}\tH@1 {metrics["hits@1"]:.6f}\tH@3 {metrics["hits@3"]:.6f}\t' \
        f'H@5 {metrics["hits@5"]:.6f}\tH@10 {metrics["hits@10"]:.6f}'

def decode(vector: Tensor, kernel: BaseKernel,
           predicate_embeddings: nn.Module) -> Tuple[int, float]:
    weight = predicate_embeddings.weight
    k = kernel.pairwise(vector, weight)[0, :]
    top_idx = k.argmax(dim=0).item()
    top_score = k[top_idx].item()
    return top_idx, top_score

def show_rules(model: SimpleHoppy,
               kernel: BaseKernel,
               predicate_embeddings: nn.Embedding,
               predicate_to_idx: Dict[str, int],
               device: Optional[torch.device] = None):
    idx_to_predicate = {i: p for p, i in predicate_to_idx.items()}

    pred_idx_pair_lst = sorted(predicate_to_idx.items(), key=lambda kv: kv[1])

    for p, i in pred_idx_pair_lst:
        indices = torch.tensor([i], dtype=torch.long, device=device)

        p_emb = predicate_embeddings(indices)

        hops_lst = [p for p in model.hops_lst]

        for reformulator, is_reversed in hops_lst:
            def _to_pair(hop: Tensor) -> Tuple[str, float]:
                idx, score = decode(hop, kernel, predicate_embeddings)
                rel = idx_to_predicate[idx]
                return rel, score

            hop_tensor_lst = [hop for hop in reformulator(p_emb)]

            r_hops = [_to_pair(hop) for hop in hop_tensor_lst]
            print(p, ' ← ', ', '.join(f'({a} {b:.4f})' for a, b in r_hops), is_reversed)
    return

def create_datasets(train_path, test_path, size_val):
    percentage_of_original_train_dataset = 0.1
    percentage_of_original_dev_dataset = 0.01
    percentage_of_original_test_dataset = 0.01
    
    split_index = round(size_val * 30000)
    output_file_names_list = []

    for dataset_name in ["train", "dev", "test"]:
        output_file_name = "{}.txt".format(dataset_name)
        if dataset_name == "train":
            with open(train_path) as f:
                entries = f.readlines()
            start = split_index
            end = split_index + round(percentage_of_original_train_dataset * ((1-size_val)*30000))
            write_dataset_name = "train"
        elif dataset_name == "dev":
            with open(train_path) as f:
                entries = f.readlines()
            start = 0
            end = round(percentage_of_original_dev_dataset * split_index)
            write_dataset_name = "train"
        elif dataset_name == "test":
            with open(test_path) as f:
                entries = f.readlines()
            start = 0
            end = round(percentage_of_original_test_dataset * 5000)
            write_dataset_name = "test"

        dataset = []

        for i in range(start, end):
            index_digit_1 = write_dataset_name + "-" + entries[i].split(" ")[0]
            index_digit_2 = write_dataset_name + "-" + entries[i].split(" ")[1]
            sum = int(entries[i].split(" ")[2])
            dataset.append((index_digit_1, sum, index_digit_2))

        write_to_file(dataset, output_file_name)
        output_file_names_list.append(output_file_name)

    return tuple(output_file_names_list)

def write_to_file(dataset, filename):   
    with open(filename, "w+") as f:
        for entry in dataset:
            for elem in entry:
                f.write(str(elem))
                f.write(" ")
            f.write("\n")

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger('nmslib').setLevel(logging.WARNING)

############################################### PARAMETERS ##############################################
nb_epochs = 1
batch_size = 16
learning_rate = 0.001
use_dropout = False
size_val = 0.1
embedding_size = 10
k_max = 4
optimizer_name = "adam"
input_type = "standard"
is_quiet = False
hops_str = ['0', '2', '1R']
nb_neg = 3
reformulator_type = "linear"
nb_rules = 0
init_type = "random"
refresh_interval = 100
index_type = "nms"
test_i_path = None
test_ii_path = None
N2_weight = None
N3_weight = None
init_size = 1.0
ref_init_type = "random"
load_path = None
save_path = None
lower_bound = -1.0
upper_bound = 1.0
is_show = False
is_fix_predicates = True
early_stopping = False
#########################################################################################################

for seed in range(0, 10):
    # setting seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    random_state = np.random.RandomState(seed)
    rs = np.random.RandomState(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # shuffle dataset and create dataset files
    generate_dataset(seed)
    train_path, dev_path, test_path = create_datasets("../data/MNIST/processed/train.txt",
        "../data/MNIST/processed/test.txt", size_val)
    
    # generate name of file that holds the trained model
    model_file_name = "NTP_final_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
        seed, nb_epochs, batch_size, learning_rate, use_dropout,
        size_val, embedding_size, k_max, optimizer_name, input_type,
        is_quiet, hops_str, nb_neg, reformulator_type, nb_rules,
        init_type, refresh_interval, index_type, test_i_path,
        test_ii_path, N2_weight, N3_weight, init_size, ref_init_type,
        load_path, save_path, lower_bound, upper_bound, is_show,
        is_fix_predicates, early_stopping)

    data = Data_MNIST(train_path=train_path, dev_path=dev_path, test_path=test_path,
                test_i_path=test_i_path, test_ii_path=test_ii_path, input_type=input_type)

    entity_embeddings = CustomEntityEmbeddings(data.idx_to_entity, use_dropout)
    predicate_embeddings = nn.Embedding(data.nb_predicates, embedding_size, sparse=True)

    if init_type in {'uniform'}:
        nn.init.uniform_(predicate_embeddings.weight, lower_bound, upper_bound)

    nn.init.uniform_(predicate_embeddings.weight, lower_bound, upper_bound)

    predicate_embeddings.weight.data *= init_size

    if is_fix_predicates is True:
        predicate_embeddings.weight.requires_grad = False

    kernel = GaussianKernel(slope=1.0)

    fact_rel = torch.tensor([data.predicate_to_idx[p] for (_, p, _) in data.train_triples],
                            dtype=torch.long, device=device)
    fact_arg1 = torch.tensor([data.entity_to_idx[s] for (s, _, _) in data.train_triples],
                                dtype=torch.long, device=device)
    fact_arg2 = torch.tensor([data.entity_to_idx[o] for (_, _, o) in data.train_triples],
                                dtype=torch.long, device=device)
    facts = [fact_rel, fact_arg1, fact_arg2]

    base_model = NeuralKB(custom_entity_embeddings=True, entity_embeddings=entity_embeddings, 
                            predicate_embeddings=predicate_embeddings,
                            k=k_max, facts=facts, kernel=kernel, device=device,
                            index_type=index_type, refresh_interval=refresh_interval).to(device)

    memory: Dict[int, MemoryReformulator.Memory] = {}

    def make_hop(s: str) -> Tuple[BaseReformulator, bool]:
        if s.isdigit():
            nb_hops, is_reversed = int(s), False
        else:
            nb_hops, is_reversed = int(s[:-1]), True
        res = None
        if reformulator_type in {'static'}:
            res = StaticReformulator(nb_hops, embedding_size, init_name=ref_init_type,
                                        lower_bound=lower_bound, upper_bound=upper_bound)
        elif reformulator_type in {'linear'}:
            res = LinearReformulator(nb_hops, embedding_size, init_name=ref_init_type,
                                        lower_bound=lower_bound, upper_bound=upper_bound)
        elif reformulator_type in {'attentive'}:
            res = AttentiveReformulator(nb_hops, predicate_embeddings, init_name=ref_init_type,
                                        lower_bound=lower_bound, upper_bound=upper_bound)
        elif reformulator_type in {'memory'}:
            if nb_hops not in memory:
                memory[nb_hops] = MemoryReformulator.Memory(nb_hops, nb_rules, embedding_size, init_name=ref_init_type)

            res = MemoryReformulator(memory[nb_hops])
        elif reformulator_type in {'ntp'}:
            res = NTPReformulator(nb_hops=nb_hops, embedding_size=embedding_size,
                                    kernel=kernel, init_name=ref_init_type,
                                    lower_bound=lower_bound, upper_bound=upper_bound)
        assert res is not None
        return res, is_reversed

    hops_lst = [make_hop(s) for s in hops_str]

    # model = MultiHoppy(model=base_model, entity_embeddings=entity_embeddings, hops_lst=hops_lst).to(device)
    # model = SimpleHoppy(model=base_model, entity_embeddings=entity_embeddings, hops_lst=hops_lst).to(device)
    model = SimpleHoppy(model=base_model, entity_embeddings=entity_embeddings, hops_lst=hops_lst).to(device)

    def scoring_function(batch_xs: np.ndarray,
                            batch_xp: np.ndarray,
                            batch_xo: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor_xs = torch.tensor(batch_xs, dtype=torch.long, device=device)
            tensor_xp = torch.tensor(batch_xp, dtype=torch.long, device=device)
            tensor_xo = torch.tensor(batch_xo, dtype=torch.long, device=device)

            tensor_xs_emb = entity_embeddings.get_logits(tensor_xs)
            tensor_xp_emb = predicate_embeddings(tensor_xp)
            tensor_xo_emb = entity_embeddings.get_logits(tensor_xo)

            scores_ = model.score(tensor_xp_emb, tensor_xs_emb, tensor_xo_emb)
        return scores_.cpu().numpy()

    # print('Model Params:', [p.shape for p in model.parameters()])
    # params_lst = {p for p in model.parameters()} | \
    #              ({entity_embeddings.weight} if is_fix_entities is False else set()) | \
    #              ({predicate_embeddings.weight} if is_fix_predicates is False else set())

    params_lst = {p for p in model.parameters()} | \
                    ({predicate_embeddings.weight} if is_fix_predicates is False else set()) | \
                    ({x for x in entity_embeddings.neural_net.parameters()})

    params = nn.ParameterList(params_lst).to(device)

    if load_path is not None:
        model.load_state_dict(torch.load(load_path))

    for tensor in params_lst:
        logger.info(f'\t{tensor.size()}\t{tensor.device}')

    optimizer_factory = {
        'adagrad': lambda arg: optim.Adagrad(arg, lr=learning_rate),
        'adam': lambda arg: optim.Adam(arg, lr=learning_rate),
        'sgd': lambda arg: optim.SGD(arg, lr=learning_rate)
    }

    assert optimizer_name in optimizer_factory
    optimizer = optimizer_factory[optimizer_name](params)
    loss_function = nn.BCELoss()

    N2_reg = N2() if N2_weight is not None else None
    N3_reg = N3() if N3_weight is not None else None

    training_time = 0
    best_accuracy = -1
    counter = 0
    for epoch_no in range(1, nb_epochs + 1):
        start_time = time.time()
        batcher = Batcher(data, batch_size, 1, random_state)
        nb_batches = len(batcher.batches)

        epoch_loss_values = []
        for batch_no, (batch_start, batch_end) in enumerate(batcher.batches, 1):
            print(f'Epoch: {epoch_no}/{nb_epochs} -- Batch: {batch_no}/{nb_batches}')
            xp_batch_np, xs_batch_np, xo_batch_np, xi_batch_np = batcher.get_batch(batch_start, batch_end)
            # xs_batch_emb = entity_embeddings.get_logits_training(xs_batch_np)
            # xo_batch_emb = entity_embeddings.get_logits_training(xo_batch_np)
            
            t = xp_batch_np.shape[0]

            assert nb_neg > 0

            xp_exp_np = np.repeat(xp_batch_np, nb_neg * 3 + 1)
            xs_exp_np = np.repeat(xs_batch_np, nb_neg * 3 + 1)
            xo_exp_np = np.repeat(xo_batch_np, nb_neg * 3 + 1)
            xi_exp_np = np.repeat(xi_batch_np, nb_neg * 3 + 1)

            xt_exp_np = np.zeros_like(xp_exp_np)
            xt_exp_np[0::nb_neg * 3 + 1] = 1

            for i in range(t):
                a_ = rs.permutation(data.nb_entities)
                b_ = rs.permutation(data.nb_entities)

                c_ = rs.permutation(data.nb_entities)
                d_ = rs.permutation(data.nb_entities)

                while a_.shape[0] < nb_neg:
                    a_ = np.concatenate([a_, rs.permutation(data.nb_entities)])
                    b_ = np.concatenate([b_, rs.permutation(data.nb_entities)])

                    c_ = np.concatenate([c_, rs.permutation(data.nb_entities)])
                    d_ = np.concatenate([d_, rs.permutation(data.nb_entities)])

                a = a_[:nb_neg]
                b = b_[:nb_neg]
                c = c_[:nb_neg]
                d = d_[:nb_neg]

                xs_exp_np[(i * nb_neg * 3) + i + 1:(i * nb_neg * 3) + nb_neg + i + 1] = a
                xo_exp_np[(i * nb_neg * 3) + nb_neg + i + 1:(i * nb_neg * 3) + nb_neg * 2 + i + 1] = b

                xs_exp_np[(i * nb_neg * 3) + nb_neg * 2 + i + 1:(i * nb_neg * 3) + nb_neg * 3 + i + 1] = c
                xo_exp_np[(i * nb_neg * 3) + nb_neg * 2 + i + 1:(i * nb_neg * 3) + nb_neg * 3 + i + 1] = d

            xp_batch = torch.tensor(xp_exp_np, dtype=torch.long, device=device)
            xs_batch = torch.tensor(xs_exp_np, dtype=torch.long, device=device)
            xo_batch = torch.tensor(xo_exp_np, dtype=torch.long, device=device)
            xi_batch = torch.tensor(xi_exp_np, dtype=torch.long, device=device)
            xt_batch = torch.tensor(xt_exp_np, dtype=torch.float32, device=device)

            # Disable masking
            # xi_batch = None

            xp_batch_emb = predicate_embeddings(xp_batch)           
            xs_batch_emb = entity_embeddings.get_logits(xs_batch)
            xo_batch_emb = entity_embeddings.get_logits(xo_batch)

            factors = [model.factor(e) for e in [xp_batch_emb, xs_batch_emb, xo_batch_emb]]
            scores = model.score(xp_batch_emb, xs_batch_emb, xo_batch_emb, mask_indices=xi_batch)          

            loss = loss_function(scores, xt_batch)
            loss += N2_weight * N2_reg(factors) if N2_weight is not None else 0.0
            loss += N3_weight * N3_reg(factors) if N3_weight is not None else 0.0

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_value = loss.item()
            epoch_loss_values += [loss_value]

            if not is_quiet:
                logger.info(f'Epoch {epoch_no}/{nb_epochs}\tBatch {batch_no}/{nb_batches}\tLoss {loss_value:.6f}')

        loss_mean, loss_std = np.mean(epoch_loss_values), np.std(epoch_loss_values)
        logger.info(f'Epoch {epoch_no}/{nb_epochs}\tLoss {loss_mean:.4f} ± {loss_std:.4f}')

        training_time += time.time() - start_time

        if early_stopping:
            dev_accuracy = evaluate_on_mnist(dev_path, data.predicate_to_idx, data.entity_to_idx, scoring_function)
            print(f'Dev accuracy after epoch {epoch_no}: {dev_accuracy}')

            if dev_accuracy > best_accuracy:
                best_accuracy = dev_accuracy
                torch.save(model.state_dict(), "current_best_model")
                counter = 0
            else:
                if counter >= 1:
                    break
                counter += 1

    if early_stopping:
        # early stopping: load best model and delete file
        model.load_state_dict(torch.load("current_best_model"))
        os.remove("current_best_model")

    # save trained model to a file
    torch.save(model.state_dict(), "results/final/{}".format(model_file_name))

    # evaluation
    start_time = time.time()
    accuracy = evaluate_on_mnist(dev_path, data.predicate_to_idx, data.entity_to_idx, scoring_function)
    testing_time = time.time() - start_time
    print('Final accuracy: {}'.format(accuracy))

    if is_show is True:
        with torch.no_grad():
            show_rules(model=model, kernel=kernel, predicate_embeddings=predicate_embeddings,
                        predicate_to_idx=data.predicate_to_idx, device=device)

    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    # remove data files
    os.remove("train.txt") 
    os.remove("dev.txt") 
    os.remove("test.txt")

    # save results to a summary file
    information = {
        "algorithm": "NTP",
        "seed": seed,
        "nb_epochs": nb_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "use_dropout": use_dropout,
        "size_val": size_val,
        "embedding_size": embedding_size,
        "k_max": k_max,
        "optimizer_name": optimizer_name,
        "input_type": input_type,
        "is_quiet": is_quiet,
        "hops_str": hops_str,
        "nb_neg": nb_neg,
        "reformulator_type": reformulator_type,
        "nb_rules": nb_rules,
        "init_type": init_type,
        "refresh_interval": refresh_interval,
        "index_type": index_type,
        "test_i_path": test_i_path,
        "test_ii_path": test_ii_path,
        "N2_weight": N2_weight,
        "N3_weight": N3_weight,
        "init_size": init_size,
        "ref_init_type": ref_init_type,
        "load_path": load_path,
        "save_path": save_path,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "is_show": is_show,
        "is_fix_predicates": is_fix_predicates,
        "early_stopping": early_stopping,
        "accuracy": accuracy,
        "training_time": training_time,
        "testing_time": testing_time,
        "model_file": model_file_name
    }
    with open("results/summary_final.json", "a") as outfile:
        json.dump(information, outfile)
        outfile.write('\n')

    # print results
    print("############################################")
    print("Seed: {} \nAccuracy: {} \nTraining time: {} \nTesting time: {}".format(seed, accuracy, 
        training_time, testing_time))
    print("############################################")