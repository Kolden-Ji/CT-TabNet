from torch import nn


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
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(p=0.25)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3840, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x
