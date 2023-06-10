import torch
from torch import nn
import torch.nn.functional as F


class SimpleNet1(nn.Module):
    def __init__(self, n_in, n_mid_1, n_mid_2, n_out):
        super(SimpleNet1, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid_1)
        self.fc2 = nn.Linear(n_mid_1, n_mid_2)
        self.fc3 = nn.Linear(n_mid_2, n_out)

    def forward(self, x):
        h1 = torch.sigmoid(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        prediction = self.fc3(h2)
        return prediction


class ConvNet1(nn.Module):
    def __init__(self, n_in, n_out):
        super(ConvNet1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n_in, out_channels=32, kernel_size=(8, 8),
                               stride=(4, 4))  # (84-8)/4 +1 =76/4 +1 =20
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2))  # (20-4)/2 +1 =9
        self.maxpool2 = nn.MaxPool2d(2, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                               stride=(1, 1))  # (9-3)/1 +1 =7  7*7*64
        self.fc1 = nn.Linear(in_features=256, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=n_out)
        self.Relu = nn.ReLU()

    # inshape (batch,x,y,channel)
    def forward(self, x):
        x = self.Relu(self.conv1(x))
        x = self.Relu(self.conv2(x))
        x = self.maxpool1(x)
        x = self.Relu(self.conv3(x))
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.Relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ConvNet4CarRacing(nn.Module):
    def __init__(self, n_in, n_out):
        super(ConvNet4CarRacing, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n_in, out_channels=32, kernel_size=(8, 8),
                               stride=(4, 4))  # (96-8)/4 +1 =88/4 +1 =23
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2))  # (23-4)/2 +1 =10

        self.maxpool1 = nn.MaxPool2d(2, stride=2)  # 64*5*5
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                               stride=(1, 1))  # (5-3)/1 +1 =3  64*3*3
        self.maxpool2 = nn.MaxPool2d(2, stride=1)  # (3+2-2)/2 + 1 64*2*2
        self.fc1 = nn.Linear(in_features=256, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=n_out)
        self.Relu = nn.ReLU()
        self.Softmax = nn.Softmax()

    # inshape (batch,x,y,channel)
    def forward(self, x):
        x = self.Relu(self.conv1(x))
        # padding on 2 sides
        # x = F.pad(x, (0, 1, 0, 1), "constant", value=0)
        x = self.Relu(self.conv2(x))
        x = self.maxpool1(x)
        x = self.Relu(self.conv3(x))
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.Relu(self.fc1(x))
        x = self.fc2(x)
        x = self.Softmax(x)
        return x
