import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return torch.relu(out + x)



class PolicyValueNetwork(nn.Module):
    def __init__(self):
        super(PolicyValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(64)

        self.block1 = ResNetBlock(64)
        self.block2 = ResNetBlock(64)
        self.block3 = ResNetBlock(64)
        self.block4 = ResNetBlock(64)

        self.p1 = nn.Conv2d(64,1,kernel_size=1,stride=1,padding=0)

        self.v1 = nn.Linear(64*64, 128)
        self.v2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.norm1(self.conv1(x))
        x = torch.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        policy = self.p1(x)
        policy = torch.flatten(policy,1)

        value = torch.flatten(x, 1)
        value = self.v1(value)
        value = torch.relu(value)
        value = self.v2(value)

        return policy,value