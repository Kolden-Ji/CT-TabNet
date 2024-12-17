import torch
from preprocessing.data_preprocessing import load_and_preprocess_data
from models.cnn_feature_extractor import CNNFeatureExtractor
from models.transformer_extractor import TransformerFeatureExtractor
from utils.feature_fusion import feature_fusion
from models.tabnet_classifier import TabNetModel
from config.train_config import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def extract_features(cnn_model, transformer_model, X_tensor):
    with torch.no_grad():
        cnn_features = cnn_model(X_tensor.to(DEVICE))
        transformer_features = transformer_model(X_tensor.to(DEVICE))
    return feature_fusion(cnn_features, transformer_features).cpu().numpy()


def main():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    print("Initializing CNN model...")
    cnn_extractor = CNNFeatureExtractor(
        input_channels=CNN_PARAMS["input_channels"],
        conv1_out_channels=CNN_PARAMS["conv1_out_channels"],
        conv1_kernel_size=CNN_PARAMS["conv1_kernel_size"],
        conv2_out_channels=CNN_PARAMS["conv2_out_channels"],
        conv2_kernel_size=CNN_PARAMS["conv2_kernel_size"],
        conv3_out_channels=CNN_PARAMS["conv3_out_channels"],
        conv3_kernel_size=CNN_PARAMS["conv3_kernel_size"],
        pooling_size=CNN_PARAMS["pooling_size"],
        dropout_rate=CNN_PARAMS["dropout_rate"],
        fc_out_features=CNN_PARAMS["fc_out_features"]
    ).to(DEVICE)

    print("Initializing Transformer model...")
    transformer_extractor = TransformerFeatureExtractor(
        input_dim=TRANSFORMER_PARAMS["input_dim"],
        num_layers=TRANSFORMER_PARAMS["num_layers"],
        num_heads=TRANSFORMER_PARAMS["num_heads"],
        ff_dim=TRANSFORMER_PARAMS["ffn_dim"],
        dropout=TRANSFORMER_PARAMS["dropout"]
    ).to(DEVICE)

    print("Extracting features...")
    train_features = extract_features(cnn_extractor, transformer_extractor, X_train_tensor)
    test_features = extract_features(cnn_extractor, transformer_extractor, X_test_tensor)

    print("Training TabNet model...")
    tabnet = TabNetModel()
    tabnet.train(train_features, y_train, test_features, y_test)

    print("Evaluating model...")
    y_pred = tabnet.predict(test_features)
    print_evaluation_metrics(y_test, y_pred)

def print_evaluation_metrics(y_true, y_pred):
    print("\nModel Evaluation Metrics:")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")
    print(f"AUC:       {roc_auc_score(y_true, y_pred):.4f}")

if __name__ == "__main__":
    try:
        main()
        print("\nTraining and evaluation completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
