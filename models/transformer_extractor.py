import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerFeatureExtractor2(nn.Module):
    def __init__(self, input_dim):
        super(TransformerFeatureExtractor2, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=4, dim_feedforward=256, dropout=0.1, activation='gelu'
            ) for _ in range(2)
        ])
        self.fc = nn.Linear(input_dim, 64)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.embedding(x).unsqueeze(1)
        x = self.position_encoding(x)
        x = self.transformer(x).mean(dim=1)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=0)
        x = self.fc(x)
        return x



class TransformerFeatureExtractor(nn.Module):
    def __init__(self, input_dim, num_layers, num_heads, ff_dim, dropout):
        super(TransformerFeatureExtractor, self).__init__()
        self.embedding = nn.Linear(input_dim, input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x).mean(dim=1)
        return self.fc(x)
