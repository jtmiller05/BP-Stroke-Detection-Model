import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)


class StrokeModel(ABC):
    def __init__(self, name):
        self.name = name
        self.model = None
        logger.info(f"Initialized {self.name} model")

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    def evaluate(self, X, y):
        """Common evaluation method for all models"""
        logger.info(f"Evaluating {self.name} model")
        try:
            y_pred = self.predict(X)
            y_prob = self.predict_proba(X)

            # Handle cases where all predictions are 0 or all are 1
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, zero_division=0),
                'recall': recall_score(y, y_pred, zero_division=0),
                'f1': f1_score(y, y_pred, zero_division=0)
            }

            # Only calculate ROC AUC if we have both classes
            if len(np.unique(y)) == 2:
                metrics['roc_auc'] = roc_auc_score(y, y_prob)
            else:
                metrics['roc_auc'] = 0.0
                logger.warning("ROC AUC score could not be calculated due to only one class present")

            logger.debug(f"Evaluation metrics for {self.name}:\n{metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error evaluating {self.name} model: {str(e)}", exc_info=True)
            raise
