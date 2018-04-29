import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # (self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [4 x 6 x 14 x 14]
        # print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))  # [4 x 16 x 5 x 5]
        # print(x.shape)
        x = x.view(-1, 16 * 5 * 5) # [4 x 400]
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # [4 x 10]
        # print(x.shape)
        return x

class GAG_Net(nn.Module):
    def __init__(self):
        super(GAG_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 10, 5)
        # (self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [4 x 6 x 14 x 14]
        # print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))  # [4 x 10 x 5 x 5])
        # print(x.shape)
        x = x.view(-1, 10,  5 * 5)  # [4 x 10 x 25]
        x = F.sigmoid(x)
        # x = x.view(x.size(0), x.size(1), -1) # [4 x 10 x 25]
        mask = nn.AdaptiveAvgPool2d((10,1))
        x = mask(x) # [4 x 10 x 1]
        x = torch.squeeze(x) # [4 x 10]

        return x