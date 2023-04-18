import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.classifier =  nn.Sequential(
            nn.Linear(3703, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 6),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = x.type(torch.float32)
        x = self.classifier(x)
        return x
    
class Net_Dropout(nn.Module):
    def __init__(self):
        super(Net_Dropout, self).__init__()
        self.classifier =  nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(3703, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 6),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x