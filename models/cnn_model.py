import torch
from torch import nn


class ConvPool(nn.Module):
    def __init__(self, channels, kernel_size, stride=2):
        super(ConvPool, self).__init__()
        self.conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        out = self.conv(x)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.avg_pool(x)
        avg_out = self.fc(x1)
        x2 = self.max_pool(x)
        max_out = self.fc(x2)
        out = avg_out + max_out
        return self.sigmoid(out)


class MultiStageCNN(nn.Module):
    def __init__(self):
        super(MultiStageCNN, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=1, stride=1, padding='same'),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=1, stride=1, padding='same'),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=1, stride=1, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.stage2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.stage3 = nn.Sequential(
            nn.Conv1d(256, 384, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm1d(384),
            nn.Conv1d(384, 384, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm1d(384),
            nn.Conv1d(384, 384, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x


class MultiBlockCNN(nn.Module):
    def __init__(self):
        super(MultiBlockCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=1, stride=1, padding='same'),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=1, stride=1, padding='same'),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 384, kernel_size=1, stride=1, padding='same'),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 384, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 384, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm1d(384),
            nn.ReLU(),
        )
        self.channelAttention = ChannelAttention(384)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)
        attention1 = self.channelAttention(x1)
        attention2 = self.channelAttention(x2)
        attention3 = self.channelAttention(x3)
        x = attention1 * x1 + attention2 * x2 + attention3 * x3
        return x


class MixedCNN(nn.Module):
    def __init__(self):
        super(MixedCNN, self).__init__()
        self.multiStageCNN = MultiStageCNN()
        self.multiBlockCNN = MultiBlockCNN()
        self.channelAttention = ChannelAttention(384)
        self.avgPool = nn.AdaptiveAvgPool1d(9)
        self.dropout = nn.Dropout(p=0.3)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3456, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.multiStageCNN(x)
        x1 = self.avgPool(x1)
        x2 = self.multiBlockCNN(x)
        x2 = self.avgPool(x2)
        attention1 = self.channelAttention(x1)
        attention2 = self.channelAttention(x2)
        x = attention1 * x1 + attention2 * x2
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x
