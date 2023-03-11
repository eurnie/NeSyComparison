import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True) 
        )
        self.classifier =  nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        return x

class Net_Dropout(nn.Module):
    def __init__(self):
        super(Net_Dropout, self).__init__()
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
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        return x

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
            nn.Linear(16 * 11 * 4, 120), # the input size is larger, because images are stacked
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 19), # second parameter is 19, because 19 different results possible
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 11 * 4)
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
            nn.Linear(16 * 11 * 4, 120), # the input size is larger, because images are stacked
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 19), # second parameter is 19, because 19 different results possible
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 11 * 4)
        x = self.classifier(x)
        return x

class Net_NN_Extra(nn.Module):
    def __init__(self):
        super(Net_NN_Extra, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2*28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 19)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class Net_SL(nn.Module):
    def __init__(self):
        super(Net_SL, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True) 
        )
        self.classifier =  nn.Sequential(
            nn.Linear(16 * 11 * 4, 120), # the input size is larger, because images are stacked
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 20), # second parameter is 20, because label of 2 images are predicted
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 11 * 4)
        x = self.classifier(x)
        return x

class Net_SL_Dropout(nn.Module):
    def __init__(self):
        super(Net_SL_Dropout, self).__init__()
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
            nn.Linear(16 * 11 * 4, 120), # the input size is larger, because images are stacked
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 20), # second parameter is 20, because label of 2 images are predicted
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 11 * 4)
        x = self.classifier(x)
        return x