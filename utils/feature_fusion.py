import torch

def feature_fusion(cnn_features, transformer_features):
    return torch.cat((cnn_features, transformer_features), dim=1)
