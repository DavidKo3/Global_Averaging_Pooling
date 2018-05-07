import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        # print(x)
        return x

class GAG_Net(nn.Module):
    def __init__(self):
        super(GAG_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 10, 5)
        self.conv2_bn = nn.BatchNorm2d(10)
        # (self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        self.fc = nn.Linear(10,10)

    def forward(self, x):
        # print(type(self.conv1))
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))  # [4 x 6 x 14 x 14]
        # print(x.shape)
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))  # [4 x 10 x 5 x 5])
        # print(x.shape)
        # x = x.view(-1, 10,  5 * 5)  # [4 x 10 x 25]

        # x = x.view(x.size(0), x.size(1), -1) # [4 x 10 x 25]
        mask = nn.AdaptiveAvgPool2d(1) # [ 5 x 5 ] -> [1 x 1]
        x = mask(x) # [4 x 10 x 1 x 1]
        # x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        # x = torch.squeeze(x) # [4 x 10]
        x = x.view(-1, 10) # [4 x 10 x 1 x 1 ] -> [4 x 10]
        x = F.relu(self.fc(x))
        # print(x)
        return x

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            size = m.weight.size()
            fan_out = size[0]  # number of rows
            fan_in = size[1]  # number of columns
            variance = np.sqrt(2.0 / (fan_in + fan_out))
            m.weight.data.normal_(0.0, variance)
            m.bias.data.normal_(0.0, variance)

        if isinstance(m, nn.Conv2d):
            size = m.weight.size()
            fan_out = size[0]  # number of rows
            fan_in = size[1]  # number of columns
            variance = np.sqrt(2.0 / (fan_in + fan_out))
            m.weight.data.normal_(0.0, variance)
            m.bias.data.normal_(0.0, variance)


class FineTuneModel(nn.Module):
    def __init__(self, original_model, arch, num_classes, num_layer) :
        super(FineTuneModel, self).__init__()

        if arch.startswith('alexnet') :
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'alexnet'
        elif arch.startswith('resnet') :
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(512, num_classes)
            )
            self.modelName = 'resnet'
        elif arch.startswith('vgg16'):
            mod = list(original_model.features.children())[:-num_layer]
            mod.append(nn.Conv2d(512, 10, 3, stride=1, padding=1))
            mod.append(nn.AdaptiveAvgPool2d(1))
            new_feature = nn.Sequential(*mod)
            original_model.features = new_feature

            self.features = original_model.features

            self.classifier= nn.Sequential(
                nn.Linear(10, 10)
            )
            self.modelName = 'vgg16'
        else :
            raise("Finetuning not supported on this architecture yet")


        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False



    def forward(self, x):
        f = self.features(x)  # [ # x 10 x 1 x 1]

        # print("feature dimension : ", f.shape)

        if self.modelName == 'alexnet':
            f = f.view(f.size(0), 256*6*6)
        elif self.modelName == 'vgg16':
            f = f.view(f.size(0), -1)

        elif self.modelName == 'resnet':
            f = f.view(f.size(0), -1)

        y = F.relu(self.classifier(f))

        return y


    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            size = m.weight.size()
            fan_out = size[0]  # number of rows
            fan_in = size[1]  # number of columns
            variance = np.sqrt(2.0 / (fan_in + fan_out))
            m.weight.data.normal_(0.0, variance)
            m.bias.data.normal_(0.0, variance)

        if isinstance(m, nn.Conv2d):
            size = m.weight.size()
            fan_out = size[0]  # number of rows
            fan_in = size[1]  # number of columns
            variance = np.sqrt(2.0 / (fan_in + fan_out))
            m.weight.data.normal_(0.0, variance)