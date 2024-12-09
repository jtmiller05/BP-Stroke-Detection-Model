import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
import optuna
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from model_base import StrokeModel

logger = logging.getLogger(__name__)


class StrokeKNN(StrokeModel):
    def __init__(self, params=None):
        super().__init__("KNN")

        self.params = params or {
            'n_neighbors': 5,
            'weights': 'uniform',
            'p': 2
        }

        self.model = KNeighborsClassifier(**self.params)
        logger.debug("KNN classifier created successfully")

    def _cv_score(self, X, y, params, n_splits=5):
        """
        Perform k-fold cross-validation and return the average F1-score.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            # Create a model with the given params
            model = KNeighborsClassifier(**params)
            model.fit(X_train_fold, y_train_fold)

            y_val_pred = model.predict(X_val_fold)
            fold_f1 = f1_score(y_val_fold, y_val_pred, zero_division=0)
            fold_scores.append(fold_f1)

        return np.mean(fold_scores)

    def tune_hyperparameters(self, X, y, n_trials=50, n_splits=5):
        """
        Use Optuna to tune hyperparameters using k-fold cross-validation on the given dataset.
        We'll optimize for F1-score.
        """
        logger.info("Starting hyperparameter tuning with Optuna for KNN using k-fold CV")

        def objective(trial):
            param = {
                'n_neighbors': trial.suggest_int('n_neighbors', 1, 50),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'p': trial.suggest_int('p', 1, 2)  # 1 = Manhattan, 2 = Euclidean
            }

            f1_avg = self._cv_score(X, y, param, n_splits=n_splits)
            return f1_avg

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best F1 score from CV: {study.best_value:.4f}")

        # Update model parameters with best found parameters
        self.params.update(study.best_params)
        self.model = KNeighborsClassifier(**self.params)
        logger.info("Model parameters updated with best hyperparameters from Optuna")

    def train(self, X, y, X_val=None, y_val=None):
        """
        Train the KNN model. KNN does not use early stopping or validation sets inherently.
        We simply fit on the provided training set.
        """
        logger.info("Starting KNN model training")
        self.model.fit(X, y)
        logger.info("KNN training completed")

    def predict(self, X):
        logger.debug(f"Making predictions for {X.shape[0]} samples")
        predictions = self.model.predict(X)
        return predictions

    def predict_proba(self, X):
        """
        KNN classifier from sklearn doesn't provide native class probabilities unless weights='distance'.
        If weights='distance', the probability can be inferred by neighbor distances. If weights='uniform',
        we approximate probability as (count of positive neighbors / k).

        For a more accurate probability estimate, consider using weights='distance'.
        """
        logger.debug(f"Computing prediction probabilities for {X.shape[0]} samples")

        # Since KNeighborsClassifier does not have predict_proba() by default for uniform weighting,
        # we will simulate probability by counting neighbors belonging to the positive class.
        # If 'weights' is 'distance', then KNeighborsClassifier will have predict_proba() available.
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        else:
            # Manual probability approximation for uniform weighting:
            neighbors = self.model.kneighbors(X, return_distance=False)
            # Retrieve classes of neighbors
            neighbor_classes = self.model._y[neighbors]
            # Probability = count of positive neighbors / number_of_neighbors
            positive_class = 1  # Assuming binary classification with classes {0,1}
            probs = np.mean(neighbor_classes == positive_class, axis=1)
            return probs
