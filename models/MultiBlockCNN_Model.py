from torch import nn


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
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


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
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 384, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 384, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.channelAttention = ChannelAttention(384)
        self.maxPool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.dropout = nn.Dropout(p=0.25)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2304, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)
        attention1 = self.channelAttention(x1)
        attention2 = self.channelAttention(x2)
        attention3 = self.channelAttention(x3)
        x = attention1 * x1 + attention2 * x2 + attention3 * x3
        x = self.maxPool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x
