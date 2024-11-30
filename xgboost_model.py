import logging

import numpy as np
import xgboost as xgb

from model_base import StrokeModel

logger = logging.getLogger(__name__)


class StrokeXGBoost(StrokeModel):
    def __init__(self, params=None):
        super().__init__("XGBoost")

        # Create early stopping callback
        early_stopping = xgb.callback.EarlyStopping(
            rounds=10,
            save_best=True,
            metric_name='auc',  # Changed to AUC for better handling of imbalanced data
            min_delta=1e-4
        )

        # Create evaluation monitor callback
        eval_monitor = xgb.callback.EvaluationMonitor(period=10)

        self.params = params or {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'early_stopping_rounds': 10,
            'callbacks': [early_stopping, eval_monitor],
            'scale_pos_weight': 1.0,  # Will be updated during training
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist'  # More efficient tree method
        }

        logger.info("Initializing XGBoost model")
        logger.debug(f"Model parameters: {self.params}")

        try:
            self.model = xgb.XGBClassifier(**self.params)
            logger.debug("XGBoost classifier created successfully")
        except Exception as e:
            logger.error(f"Failed to initialize XGBoost model: {str(e)}", exc_info=True)
            raise

    def train(self, X, y):
        logger.info("Starting XGBoost model training")
        logger.debug(f"Training data shape: X={X.shape}, y={y.shape}")

        try:
            # Calculate class weights
            neg_count = (y == 0).sum()
            pos_count = (y == 1).sum()
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

            # Update scale_pos_weight parameter
            self.model.set_params(scale_pos_weight=scale_pos_weight)
            logger.info(f"Set scale_pos_weight to {scale_pos_weight} based on class distribution")

            # Train the model with evaluation set for early stopping
            self.model.fit(
                X, y,
                eval_set=[(X, y)],
                verbose=True
            )

            logger.info("XGBoost training completed")
            feature_importance = dict(zip(X.columns, self.model.feature_importances_))
            logger.debug(f"Feature importances: {feature_importance}")

        except Exception as e:
            logger.error(f"Error during XGBoost training: {str(e)}", exc_info=True)
            raise

    def predict(self, X):
        logger.debug(f"Making predictions for {X.shape[0]} samples")
        try:
            predictions = self.model.predict(X)
            prediction_counts = np.bincount(predictions, minlength=2)
            logger.debug(f"Prediction distribution: {prediction_counts}")
            return predictions
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            raise

    def predict_proba(self, X):
        logger.debug(f"Computing prediction probabilities for {X.shape[0]} samples")
        try:
            probabilities = self.model.predict_proba(X)[:, 1]
            logger.debug(f"Probability statistics - Mean: {np.mean(probabilities):.4f}, "
                         f"Std: {np.std(probabilities):.4f}")
            return probabilities
        except Exception as e:
            logger.error(f"Error during probability prediction: {str(e)}", exc_info=True)
            raise
