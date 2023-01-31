#ref:https://github.com/TadaoYamaoka/python-dlshogi2/blob/main/pydlshogi2/network/policy_value_resnet.py

import torch
import torch.nn as nn

#ResNetBlock:dlshogiとほぼ同等の構造。
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


#ResNetBlockを並べたネットワーク、channelとblock数を設定できる。
class PolicyValueNetwork(nn.Module):
    def __init__(self, channels=64, blocks=4, fcl=128):
        super(PolicyValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(2, channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(channels)

        self.blocks = torch.nn.Sequential(*[ResNetBlock(channels) for _ in range(blocks)])

        self.p1 = nn.Conv2d(channels,1,kernel_size=1,stride=1,padding=0)

        self.v1 = nn.Linear(channels*64, fcl)
        self.v2 = nn.Linear(fcl, 1)

    def forward(self, x):
        x = self.norm1(self.conv1(x))
        x = torch.relu(x)
        
        x = self.blocks(x)

        policy = self.p1(x)
        policy = torch.flatten(policy,1)

        value = torch.flatten(x, 1)
        value = self.v1(value)
        value = torch.relu(value)
        value = self.v2(value)

        return policy,value
