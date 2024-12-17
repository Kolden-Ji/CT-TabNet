from pytorch_tabnet.tab_model import TabNetClassifier
from config.train_config import TABNET_PARAMS, EARLY_STOPPING_PATIENCE, EPOCHS, BATCH_SIZE


class TabNetModel2:
    def __init__(self):
        self.model = TabNetClassifier(
            n_d=64, n_a=64, n_steps=5, gamma=1.5,
            n_independent=2, n_shared=2,
            momentum=0.02, virtual_batch_size=128,
            dropout=0.2
        )

    def train(self, X_train, y_train, X_test, y_test):
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            patience=5, max_epochs=50,
            batch_size=64, virtual_batch_size=128,
            loss_fn='cross_entropy'
        )

    def predict(self, X):
        return self.model.predict(X)

    def feature_importance(self):
        return self.model.feature_importances_


class TabNetModel:
    def __init__(self):
        self.model = TabNetClassifier(
            n_d=TABNET_PARAMS["n_d"],
            n_a=TABNET_PARAMS["n_a"],
            n_steps=TABNET_PARAMS["n_steps"],
            gamma=TABNET_PARAMS["gamma"],
            n_independent=TABNET_PARAMS["n_independent"],
            n_shared=TABNET_PARAMS["n_shared"],
            momentum=TABNET_PARAMS["momentum"],
            virtual_batch_size=TABNET_PARAMS["virtual_batch_size"],
            dropout=TABNET_PARAMS["dropout"]
        )

    def train(self, X_train, y_train, X_test, y_test):
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            patience=EARLY_STOPPING_PATIENCE,
            max_epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            virtual_batch_size=TABNET_PARAMS["virtual_batch_size"],
            loss_fn='cross_entropy'
        )

    def predict(self, X):
        return self.model.predict(X)

    def feature_importance(self):
        return self.model.feature_importances_