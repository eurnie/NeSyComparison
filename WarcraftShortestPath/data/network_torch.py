import torch.nn as nn

class Net_NN(nn.Module):
    def __init__(self):
        super(Net_NN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True)
        )
        self.classifier =  nn.Sequential(
            nn.Linear(7056, 3528),
            nn.ReLU(),
            nn.Linear(3528, 1764),
            nn.ReLU(),
            nn.Linear(1764, 882),
            nn.ReLU(),
            nn.Linear(882, 441),
            nn.ReLU(),
            nn.Linear(441, 144),
            nn.ReLU(),
            nn.Linear(144, 144),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 7056)
        x = self.classifier(x)
        return x
    
class Net_NN_Dropout(nn.Module):
    def __init__(self):
        super(Net_NN_Dropout, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True) 
        )
        self.classifier =  nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(16 * 11 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 19),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 11 * 4)
        x = self.classifier(x)
        return x