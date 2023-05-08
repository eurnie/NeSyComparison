import torch.nn as nn
import torch
    
class Net_CiteSeer(nn.Module):
    def __init__(self, dropout_rate):
        super(Net_CiteSeer, self).__init__()
        self.classifier =  nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(3703, 840),
            nn.ReLU(),
            nn.Linear(840, 84),
            nn.ReLU(),
            nn.Linear(84, 6),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = x.type(torch.float32)
        x = self.classifier(x)
        return x
    
class Net_Cora(nn.Module):
    def __init__(self, dropout_rate):
        super(Net_Cora, self).__init__()
        self.classifier =  nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(1433, 840),
            nn.ReLU(),
            nn.Linear(840, 84),
            nn.ReLU(),
            nn.Linear(84, 7),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = x.type(torch.float32)
        x = self.classifier(x)
        return x
    
class Net_PubMed(nn.Module):
    def __init__(self, dropout_rate):
        super(Net_PubMed, self).__init__()
        self.classifier =  nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 75),
            nn.ReLU(),
            nn.Linear(75, 3),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = x.type(torch.float32)
        x = self.classifier(x)
        return x