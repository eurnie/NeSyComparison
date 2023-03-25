import sys
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path

sys.path.append("..")
from data.network_torch import Net, Net_Dropout

class CustomEntityEmbeddings:
    def __init__(self, idx_to_entity, learning_rate, use_dropout):
        DATA_ROOT = Path(__file__).parent.parent.joinpath('data')
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        self.datasets = {
            "train": torchvision.datasets.MNIST(
                root=str(DATA_ROOT), train=True, download=True, transform=transform
            ),
            "test": torchvision.datasets.MNIST(
                root=str(DATA_ROOT), train=False, download=True, transform=transform
            ),
        }
        self.learning_rate = learning_rate
        self.idx_to_entity = idx_to_entity
        if use_dropout:
            self.neural_net = Net_Dropout()
        else:
            self.neural_net = Net()

    def ids_to_images(self, ids):
        images_list = []
        for id in ids.numpy():
            entity_name = self.idx_to_entity[id]
            dataset_name = entity_name.split("-")[0]
            mnist_index = int(entity_name.split("-")[1])
            image = self.datasets[dataset_name][mnist_index][0].numpy()
            images_list.append(image)
        return torch.from_numpy(np.array(images_list))
    
    def get_logits(self, ids):
        return self.neural_net(self.ids_to_images(ids))