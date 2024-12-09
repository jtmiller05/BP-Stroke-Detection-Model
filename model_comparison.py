import logging

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class ModelComparison:
    def __init__(self, models, X, y):
        self.models = models
        logger.info(f"Initializing model comparison with {len(models)} models")

        logger.debug("Splitting data into train and test sets")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Log initial class distribution in the smaller training set
        self._log_distribution("Initial training set (pre-SMOTE)", self.y_train)

        # Apply SMOTE to the training data
        smote = SMOTE(random_state=50)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

        # Log class distribution after SMOTE
        self._log_distribution("After SMOTE", self.y_train)

        # Visualize the distributions
        # self._plot_distribution_comparison()

        logger.info(f"Train set size: {self.X_train.shape}, Test set size: {self.X_test.shape}")
        self.results = {}

    def _log_distribution(self, label, y):
        """Log the class distribution with counts and percentages"""
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        distribution = pd.DataFrame({
            'Class': unique,
            'Count': counts,
            'Percentage': (counts / total) * 100
        })

        logger.info(f"{label} class distribution:")
        for _, row in distribution.iterrows():
            logger.info(f"Class {int(row['Class'])}: {int(row['Count'])} samples "
                        f"({row['Percentage']:.2f}%)")

    def train_and_evaluate(self):
        logger.info("Starting model training and evaluation")

        for model in self.models:
            logger.info(f"Processing {model.name}")

            try:
                # If XGBoost, run hyperparameter tuning on the train/validation split
                # if model.name == 'XGBoost':
                #     model.tune_hyperparameters(self.X_train, self.y_train, n_trials=50)

                # Training using the entire training set (post-SMOTE)
                logger.debug(f"Training {model.name}")
                model.train(self.X_train, self.y_train)

                # Evaluation
                logger.debug(f"Evaluating {model.name} on train set")
                train_metrics = model.evaluate(self.X_train, self.y_train)

                logger.debug(f"Evaluating {model.name} on test set")
                test_metrics = model.evaluate(self.X_test, self.y_test)

                self.results[model.name] = {
                    'train': train_metrics,
                    'test': test_metrics
                }

                logger.info(f"Completed evaluation for {model.name}")

            except Exception as e:
                logger.error(f"Error processing {model.name}: {str(e)}", exc_info=True)
                raise

    def print_results(self):
        logger.info("Printing model comparison results")

        for model_name, metrics in self.results.items():
            logger.info(f"\n{model_name} Results:")
            logger.info("Train Metrics:")
            for metric, value in metrics['train'].items():
                logger.info(f"{metric}: {value:.4f}")
            logger.info("\nTest Metrics:")
            for metric, value in metrics['test'].items():
                logger.info(f"{metric}: {value:.4f}")
