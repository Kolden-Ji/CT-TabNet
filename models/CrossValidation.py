import numpy as np
import pandas as pd
import torch
# from imblearn.over_sampling import RandomOverSampler
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import KFold, StratifiedKFold
from torch import nn, optim
from torch.optim import lr_scheduler

from MultiStageCNN_Model import MultiStageCNN
from config.train_config import DATA_PATH

# from MultiBlockCNN_Model import MultiBlockCNN
# from CNN_Model_v2 import MixedCNN
# DATA_FILE = '../0_DATA/diabetesData/yjs_dataset_ver3.xlsx'
# DATA_FILE = '../0_DATA/diabetesData/diabetes_data_Bangladesh.csv'
# DATA_FILE = '../0_DATA/irisData/iris_train_data.csv'
# DATA_FILE = '../0_DATA/heartDiseaseData/trainData.csv'

NUM_EPOCHS = 100
BATCH_SIZE = 300


def rand_index(args_data, args_label):
    rows, cols = args_data.size()

    random_indices = torch.randperm(rows)

    return args_data[random_indices], args_label[random_indices]


def model_predict(model, x_test):
    model.eval()
    with torch.no_grad():
        test_data = x_test.unsqueeze(1)
        outputs = model(test_data)
        _, predicted = torch.max(outputs.data, 1)
    return predicted


def fit_model(x, y, x_test, y_test):
    model = MultiStageCNN().to('cuda')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)    # 0.0005

    num_epochs = NUM_EPOCHS
    batch_size = BATCH_SIZE

    x, y = rand_index(x, y)

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for i in range(0, len(x), batch_size):
            inputs = x[i:i + batch_size]
            inputs = inputs.unsqueeze(1)
            labels = y[i:i + batch_size]

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

        # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / total_samples:.4f}")

        y_pred = model_predict(model, x_test)
        accuracy = accuracy_score(y_test.cpu().numpy(), y_pred.cpu().numpy())
        if best_acc < accuracy:
            torch.save(model.state_dict(),
                       '')
            best_acc = accuracy
            best_epoch = epoch
    print(f'best_epoch:{best_epoch}, best_acc:{best_acc:.4f}')


def main():
    data = pd.read_csv(DATA_PATH)

    labels = data['Outcome']
    features = data.drop(['Outcome'], axis=1)

    features = np.array(features)
    labels = np.array(labels)

    stratifiedKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=385)

    train_accuracies = []

    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    aucs = []
    num = 0
    for train_indices, test_indices in stratifiedKFold.split(features, labels):
        print(f'第{num + 1}轮实验')
        num = num + 1

        x_train, y_train = features[train_indices], labels[train_indices]
        x_test, y_test = features[test_indices], labels[test_indices]

        x_train = torch.Tensor(x_train)
        y_train = torch.LongTensor(y_train)
        x_test = torch.Tensor(x_test)
        y_test = torch.LongTensor(y_test)
        device = torch.device("cuda")
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        fit_model(x_train, y_train, x_test, y_test)

        best_model = MultiStageCNN().to('cuda')
        best_model.load_state_dict(torch.load(
            ''))
        best_model.eval()

        y_train_pred = model_predict(best_model, x_train)
        train_accuracy = accuracy_score(y_train.cpu().numpy(), y_train_pred.cpu().numpy())
        train_accuracies.append(train_accuracy)

        y_pred = model_predict(best_model, x_test)
        accuracy = accuracy_score(y_test.cpu().numpy(), y_pred.cpu().numpy())
        precision = precision_score(y_test.cpu().numpy(), y_pred.cpu().numpy())
        recall = recall_score(y_test.cpu().numpy(), y_pred.cpu().numpy())
        f1 = f1_score(y_test.cpu().numpy(), y_pred.cpu().numpy())
        fpr, tpr, thresholds = roc_curve(y_test.cpu().numpy(), y_pred.cpu().numpy())
        auc_value = auc(fpr, tpr)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        aucs.append(auc_value)

    all_result = pd.DataFrame({'Accuracy': accuracies,
                               'Precision': precisions,
                               'Recall': recalls,
                               'F1': f1s,
                               'AUC': aucs})
    print(all_result)

    print(f"平均准确率: {np.mean(accuracies):.4f} 准确率标准差: {np.std(accuracies):.4f}")
    print(f"平均精准率: {np.mean(precisions):.4f} 精准率标准差: {np.std(precisions):.4f}")
    print(f"平均召回率: {np.mean(recalls):.4f} 召回率标准差: {np.std(recalls):.4f}")
    print(f"平均F1值: {np.mean(f1s):.4f} F1值标准差: {np.std(f1s):.4f}")
    print(f"平均AUC: {np.mean(aucs):.4f} AUC标准差: {np.std(aucs):.4f}")

    x_axis = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plt.ylim(0.78, 1.005)
    plt.plot(x_axis, train_accuracies, label='Training set accuracy')
    plt.plot(x_axis, accuracies, label='Test set accuracy')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Train and Test set accuracy')

    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()


