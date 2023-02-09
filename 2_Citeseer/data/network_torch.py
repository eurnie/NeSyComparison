import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3703, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return F.softmax(logits, dim=0)

class Net_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3703, 16)
        self.conv2 = GCNConv(16, 6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)