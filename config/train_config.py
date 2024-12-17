EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_DELTA = 0.001

OPTIMIZER = "AdamW"
SCHEDULER = "CosineAnnealingLR"

CNN_PARAMS = {
    "input_channels": 1,
    "conv1_out_channels": 32,
    "conv1_kernel_size": 3,
    "conv2_out_channels": 32,
    "conv2_kernel_size": 5,
    "conv3_out_channels": 32,
    "conv3_kernel_size": 7,
    "pooling_size": 2,
    "dropout_rate": 0.3,
    "fc_out_features": 64
}

TRANSFORMER_PARAMS = {
    "input_dim": 64,
    "num_layers": 2,
    "num_heads": 4,
    "ffn_dim": 256,
    "dropout": 0.1
}


TABNET_PARAMS = {
    "n_d": 64,
    "n_a": 64,
    "n_steps": 5,
    "gamma": 1.5,
    "n_independent": 2,
    "n_shared": 2,
    "momentum": 0.02,
    "virtual_batch_size": 128,
    "dropout": 0.2
}

METRICS = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]

DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"

DATA_PATH = "./diabetes.xlsx"

