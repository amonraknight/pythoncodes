import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, n_in, n_mid_1, n_mid_2, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid_1)
        self.fc2 = nn.Linear(n_mid_1, n_mid_2)
        self.fc3 = nn.Linear(n_mid_2, n_out)

    def forward(self, x):
        h1 = torch.sigmoid(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        prediction = self.fc3(h2)
        return prediction


