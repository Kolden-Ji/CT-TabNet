import torch
from preprocessing.data_preprocessing import load_and_preprocess_data
from models.cnn_feature_extractor import CNNFeatureExtractor2
from models.transformer_extractor import TransformerFeatureExtractor2
from utils.feature_fusion import feature_fusion
from models.tabnet_classifier import TabNetModel2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

X_train, X_test, y_train, y_test = load_and_preprocess_data("D:/data/糖尿病/diabetes.xlsx")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

cnn_extractor = CNNFeatureExtractor2(input_dim=X_train.shape[1])
transformer_extractor = TransformerFeatureExtractor2(input_dim=X_train.shape[1])

with torch.no_grad():
    cnn_train = cnn_extractor(X_train_tensor)
    transformer_train = transformer_extractor(X_train_tensor)
    cnn_test = cnn_extractor(X_test_tensor)
    transformer_test = transformer_extractor(X_test_tensor)

train_features = feature_fusion(cnn_train, transformer_train).numpy()
test_features = feature_fusion(cnn_test, transformer_test).numpy()

tabnet = TabNetModel2()
tabnet.train(train_features, y_train, test_features, y_test)

y_pred = tabnet.predict(test_features)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_pred))
