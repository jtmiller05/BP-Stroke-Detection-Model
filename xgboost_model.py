import logging

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from model_base import StrokeModel

logger = logging.getLogger(__name__)


class StrokeXGBoost(StrokeModel):
    def __init__(self, params=None):
        super().__init__("XGBoost")

        # Create early stopping callback
        early_stopping = xgb.callback.EarlyStopping(
            rounds=10,
            save_best=True,
            metric_name='auc',
            min_delta=1e-4
        )

        # Create evaluation monitor callback
        eval_monitor = xgb.callback.EvaluationMonitor(period=10)

        self.params = params or {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'callbacks': [early_stopping, eval_monitor],
            'max_depth': 4,
            'learning_rate': 0.005,
            'min_child_weight': 6,
            'subsample': 0.74,
            'colsample_bytree': 0.76,
            'gamma': 0.18,
            'reg_alpha': 0.23,
            'reg_lambda': 0.76,
            'n_estimators': 100,
            'tree_method': 'hist'
        }

        self.model = xgb.XGBClassifier(**self.params)
        logger.debug("XGBoost classifier created successfully")

    def _cv_score(self, X, y, params, n_splits=5):
        """
        Perform k-fold cross-validation and return the average F1-score.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_scores = []

        # Remove callbacks from params for CV
        cv_params = params.copy()
        cv_params.pop('callbacks', None)  # Remove callbacks if present

        for train_idx, val_idx in skf.split(X, y):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            # Create a model with the given params
            model = xgb.XGBClassifier(**cv_params)

            # Fit with early stopping using validation set
            model.fit(
                X_train_fold,
                y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False
            )

            y_val_pred = model.predict(X_val_fold)
            fold_f1 = f1_score(y_val_fold, y_val_pred, zero_division=0)
            fold_scores.append(fold_f1)

        return np.mean(fold_scores)

    def tune_hyperparameters(self, X, y, n_trials=100, n_splits=10):
        """
        Use Optuna to tune hyperparameters using k-fold cross-validation on the given dataset.
        We'll optimize for F1-score.
        """
        logger.info("Starting hyperparameter tuning with Optuna using k-fold CV")

        def objective(trial):
            param = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'learning_rate': trial.suggest_float('learning_rate', 1e-2, 1e-1, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 5, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 0.8),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'tree_method': 'hist'
            }

            # Evaluate params using cross-validation
            f1_avg = self._cv_score(X, y, param, n_splits=n_splits)
            return f1_avg

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best F1 score from CV: {study.best_value:.4f}")

        # Update model parameters with best found parameters
        self.params.update(study.best_params)

        # Add callbacks back when creating the final model
        early_stopping = xgb.callback.EarlyStopping(
            rounds=10,
            save_best=True,
            metric_name='auc',
            min_delta=1e-4
        )
        eval_monitor = xgb.callback.EvaluationMonitor(period=10)
        self.params['callbacks'] = [early_stopping, eval_monitor]

        self.model = xgb.XGBClassifier(**self.params)
        logger.info("Model parameters updated with best hyperparameters from Optuna")

    def train(self, X, y, X_val=None, y_val=None):
        """
        Train the XGBoost model. If validation sets are provided, it uses them for early stopping.
        """
        logger.info("Starting XGBoost model training")

        eval_set = [(X, y)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

            # Adjust scale_pos_weight based on current distribution
            neg_count = (y == 0).sum()
            pos_count = (y == 1).sum()
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            self.model.set_params(scale_pos_weight=scale_pos_weight)
            logger.info(f"Set scale_pos_weight to {scale_pos_weight} based on class distribution")

        self.model.fit(
            X, y,
            eval_set=eval_set,
            verbose=True
        )

        logger.info("XGBoost training completed")
        feature_importance = dict(zip(X.columns, self.model.feature_importances_))
        logger.debug(f"Feature importances: {feature_importance}")

    def predict(self, X):
        logger.debug(f"Making predictions for {X.shape[0]} samples")
        predictions = self.model.predict(X)
        return predictions

    def predict_proba(self, X):
        logger.debug(f"Computing prediction probabilities for {X.shape[0]} samples")
        probabilities = self.model.predict_proba(X)[:, 1]
        return probabilities

    def evaluate(self, X, y):
        """
        Evaluate the model on a given dataset and return a dictionary of metrics.
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y, y_proba)
        accuracy = (y_pred == y).mean()

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }

    def cv_evaluate(self, X, y, n_splits=5):
        """
        Optional: Evaluate the trained model using k-fold cross-validation and return average metrics.
        This method can be called after training or hyperparameter tuning to verify stability.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        accuracies, precisions, recalls, f1s, roc_aucs = [], [], [], [], []

        for train_idx, val_idx in skf.split(X, y):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            # Retrain model on this fold
            self.model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)],
                           early_stopping_rounds=20, verbose=False)

            y_val_pred = self.model.predict(X_val_fold)
            y_val_proba = self.model.predict_proba(X_val_fold)[:, 1]

            accuracies.append((y_val_pred == y_val_fold).mean())
            precisions.append(precision_score(y_val_fold, y_val_pred, zero_division=0))
            recalls.append(recall_score(y_val_fold, y_val_pred, zero_division=0))
            f1s.append(f1_score(y_val_fold, y_val_pred, zero_division=0))
            roc_aucs.append(roc_auc_score(y_val_fold, y_val_proba))

        return {
            'accuracy': np.mean(accuracies),
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1': np.mean(f1s),
            'roc_auc': np.mean(roc_aucs)
        }
