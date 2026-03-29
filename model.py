import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_RES_BLOCKS, NUM_FILTERS, INPUT_PLANES, POLICY_SIZE


class ResBlock(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class AlphaZeroNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_input = nn.Conv2d(INPUT_PLANES, NUM_FILTERS, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(NUM_FILTERS)
        self.res_blocks = nn.ModuleList([ResBlock(NUM_FILTERS) for _ in range(NUM_RES_BLOCKS)])

        # Policy head
        self.policy_conv = nn.Conv2d(NUM_FILTERS, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 64, POLICY_SIZE)

        # Value head
        self.value_conv = nn.Conv2d(NUM_FILTERS, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(64, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            x = block(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v
