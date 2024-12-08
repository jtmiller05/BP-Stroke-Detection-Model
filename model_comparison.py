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

        # Log initial class distribution
        self._log_distribution("Initial training set", self.y_train)

        # Apply SMOTE to the training data
        smote = SMOTE(random_state=42)
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

    def _plot_distribution_comparison(self):
        """Create and save a visualization of the class distribution before and after SMOTE"""
        try:
            plt.figure(figsize=(12, 6))

            # Calculate class distributions
            orig_unique, orig_counts = np.unique(self.y_test, return_counts=True)
            smote_unique, smote_counts = np.unique(self.y_train, return_counts=True)

            # Create bar positions
            x = np.arange(len(orig_unique))
            width = 0.35

            # Create bars
            plt.bar(x - width / 2, orig_counts / sum(orig_counts) * 100,
                    width, label='Original Distribution (Test Set)',
                    color='skyblue', alpha=0.7)
            plt.bar(x + width / 2, smote_counts / sum(smote_counts) * 100,
                    width, label='After SMOTE (Train Set)',
                    color='lightcoral', alpha=0.7)

            # Customize plot
            plt.xlabel('Class')
            plt.ylabel('Percentage of Samples')
            plt.title('Class Distribution Before and After SMOTE')
            plt.xticks(x, [f'Class {int(i)}' for i in orig_unique])
            plt.legend()

            # Add percentage labels on bars
            def add_labels(x, counts):
                total = sum(counts)
                for i, count in enumerate(counts):
                    percentage = (count / total) * 100
                    plt.text(x[i], percentage, f'{percentage:.1f}%',
                             ha='center', va='bottom')

            add_labels(x - width / 2, orig_counts)
            add_labels(x + width / 2, smote_counts)

            # Save plot
            plt.tight_layout()
            plt.savefig('class_distribution.png')
            plt.close()

            logger.info("Class distribution plot saved as 'class_distribution.png'")

        except Exception as e:
            logger.error(f"Error creating distribution plot: {str(e)}", exc_info=True)

    def train_and_evaluate(self):
        logger.info("Starting model training and evaluation")

        for model in self.models:
            logger.info(f"Processing {model.name}")

            try:
                # Training
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
