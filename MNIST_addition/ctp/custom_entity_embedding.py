import sys
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path
from torch import nn

sys.path.append("..")
from data.network_torch import Net_NTP

class CustomEntityEmbeddings:
    def __init__(self, dataset, idx_to_entity, use_dropout):
        DATA_ROOT = Path(__file__).parent.parent.joinpath('data')
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        if dataset == "mnist":
            self.datasets = {
                "train": torchvision.datasets.MNIST(
                    root=str(DATA_ROOT), train=True, download=True, transform=transform
                ),
                "test": torchvision.datasets.MNIST(
                    root=str(DATA_ROOT), train=False, download=True, transform=transform
                ),
            }
        elif dataset == "fashion_mnist":
            self.datasets = {
                "train": torchvision.datasets.FashionMNIST(
                    root=str(DATA_ROOT), train=True, download=True, transform=transform
                ),
                "test": torchvision.datasets.FashionMNIST(
                    root=str(DATA_ROOT), train=False, download=True, transform=transform
                ),
            }
        self.idx_to_entity = idx_to_entity
        self.neural_net = Net_NTP()

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.neural_net.parameters(), lr=0.001)

    def ids_to_images(self, ids):
        images_list = []
        labels_list = []
        for id in ids:
            entity_name = self.idx_to_entity[id.item()]
            dataset_name = entity_name.split("-")[0]
            mnist_index = int(entity_name.split("-")[1])
            image = self.datasets[dataset_name][mnist_index][0].numpy()
            label = self.datasets[dataset_name][mnist_index][1]
            images_list.append(image)
            labels_list.append(label)
        return torch.from_numpy(np.array(labels_list)), torch.from_numpy(np.array(images_list))
    
    def get_logits_training(self, ids):
        self.neural_net.train()
        labels, images = self.ids_to_images(ids)
        logits = self.neural_net(images)
        loss = self.loss_fn(logits, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_logits(self, ids):
        self.neural_net.eval()
        _, images = self.ids_to_images(ids)
        return self.neural_net(images)