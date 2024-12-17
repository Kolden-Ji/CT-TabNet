import torch.nn as nn
import torch

class CNNFeatureExtractor2(nn.Module):
    def __init__(self, input_dim):
        super(CNNFeatureExtractor2, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(32)
        )
        self.fc = nn.Linear(32, 64)
        self.branch1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.dropout = nn.Dropout(0.3)
        self.residual = nn.Linear(input_dim, 64)
        self.fc = nn.Linear(32 * 3, 64)

    def forward(self, x):
        x = x.unsqueeze(1)
        branch1_out = self.branch1(x).mean(dim=-1)
        branch2_out = self.branch2(x).mean(dim=-1)
        branch3_out = self.branch3(x).mean(dim=-1)
        combined = torch.cat([branch1_out, branch2_out, branch3_out], dim=1)
        combined = self.dropout(combined)
        residual = self.residual(x.squeeze(1))
        return self.fc(combined) + residual

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels, conv1_out_channels, conv1_kernel_size,
                 conv2_out_channels, conv2_kernel_size, conv3_out_channels, conv3_kernel_size,
                 pooling_size, dropout_rate, fc_out_features):
        super(CNNFeatureExtractor, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_channels, conv1_out_channels, kernel_size=conv1_kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pooling_size),
            nn.Conv1d(conv1_out_channels, conv2_out_channels, kernel_size=conv2_kernel_size, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pooling_size),
            nn.Conv1d(conv2_out_channels, conv3_out_channels, kernel_size=conv3_kernel_size, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pooling_size),
            nn.Dropout(dropout_rate)
        )
        self.fc = nn.Linear(conv3_out_channels, fc_out_features)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layers(x)
        x = x.mean(dim=-1)
        return self.fc(x)