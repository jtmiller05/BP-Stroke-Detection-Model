import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)

class ModelComparison:
    def __init__(self, models, X, y):
        self.models = models
        logger.info(f"Initializing model comparison with {len(models)} models")

        # Apply SMOTE to the entire dataset before splitting
        logger.info("Applying SMOTE before splitting into train/test sets")
        smote = SMOTE(random_state=50)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        self._log_distribution("Post-SMOTE (entire dataset)", y_resampled)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
        )

        self._log_distribution("Final training set distribution", self.y_train)
        self._log_distribution("Final test set distribution", self.y_test)

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
                # If model requires hyperparameter tuning
                if model.name == 'KNN':
                    model.tune_hyperparameters(self.X_train, self.y_train, n_trials=300)

                # Train the model
                logger.debug(f"Training {model.name}")
                model.train(self.X_train, self.y_train)

                # Evaluate on train and test sets
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

    def generate_visuals(self, output_path='model_comparison.png'):
        """
        Generate and save visuals comparing the performances of each model.
        Shows training vs test metrics comparison and confusion matrices for each model.
        """
        logger.info("Generating comparison visuals")

        # Extract metrics for comparison
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        model_names = list(self.results.keys())

        # Build DataFrames for train and test metrics
        train_data = []
        test_data = []
        for model_name in model_names:
            train_metrics = self.results[model_name]['train']
            test_metrics = self.results[model_name]['test']
            train_row = [train_metrics.get(m, np.nan) for m in metrics_to_plot]
            test_row = [test_metrics.get(m, np.nan) for m in metrics_to_plot]
            train_data.append(train_row)
            test_data.append(test_row)

        train_df = pd.DataFrame(train_data, columns=metrics_to_plot, index=model_names)
        test_df = pd.DataFrame(test_data, columns=metrics_to_plot, index=model_names)

        n_models = len(model_names)
        n_metrics = len(metrics_to_plot)

        fig = plt.figure(figsize=(15, 12))

        # Create a grid layout
        gs = plt.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.35)

        # Top subplot for metrics comparison
        ax_metrics = fig.add_subplot(gs[0])

        # Bottom row for confusion matrices (subplots)
        # We'll create one confusion matrix per model in a row
        # We'll do this after plotting the metrics
        from sklearn.metrics import confusion_matrix

        # Parameters for grouped bar chart
        bar_width = 0.08  # width of each bar
        # Each metric will have n_models * 2 bars (train and test)
        # We'll center each cluster of bars around x = metric_index
        x = np.arange(n_metrics)

        # Use a color map for models
        cmap = plt.cm.get_cmap('tab10', n_models)
        model_colors = [cmap(i) for i in range(n_models)]

        # Plot bars
        # For each metric, we have a cluster at x[m].
        # Each cluster contains n_models * 2 bars: model i (train), model i (test)
        # We'll arrange them from left to right around x[m].
        # Total width of cluster = n_models * 2 * bar_width
        # Start from the left: x[m] - (n_models * bar_width)
        for m_idx, metric in enumerate(metrics_to_plot):
            cluster_center = x[m_idx]
            total_width = n_models * 2 * bar_width
            start = cluster_center - (n_models * bar_width)

            for i, model_name in enumerate(model_names):
                train_val = train_df.iloc[i, m_idx]
                test_val = test_df.iloc[i, m_idx]

                # Position for train bar for model i
                train_x = start + (i * 2 * bar_width)
                # Position for test bar for model i (right next to train)
                test_x = train_x + bar_width

                # Plot train bar
                ax_metrics.bar(train_x, train_val, width=bar_width,
                               color=model_colors[i],
                               label=None if m_idx > 0 else f'{model_name} (Train)',
                               alpha=0.8)

                # Plot test bar (same color, but add hatch to differentiate)
                ax_metrics.bar(test_x, test_val, width=bar_width,
                               color=model_colors[i],
                               hatch='//',
                               label=None if m_idx > 0 else f'{model_name} (Test)',
                               alpha=0.8)

        ax_metrics.set_ylabel('Score')
        ax_metrics.set_title('Model Performance Comparison (Train vs Test)', pad=20)
        ax_metrics.set_xticks(x)
        ax_metrics.set_xticklabels(metrics_to_plot)
        ax_metrics.set_ylim(0, 1.05)
        ax_metrics.grid(True, linestyle='--', alpha=0.7)

        # Create a custom legend
        # We only need one entry per model for train and one for test.
        # We'll just use the bars from the first metric cluster.
        handles, labels = ax_metrics.get_legend_handles_labels()
        # The above might have duplicates since we labeled each model_name (Train) and (Test)
        # only in the first metric, so we can filter duplicates
        seen = set()
        final_handles = []
        final_labels = []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                final_handles.append(h)
                final_labels.append(l)

        ax_metrics.legend(final_handles, final_labels, bbox_to_anchor=(1.05, 1), loc='upper left')

        # Confusion matrices
        # We'll create one row with all confusion matrices. Each model gets one subplot.
        # Adjust subplot by creating a new GridSpec from the remaining space
        gs_bottom = GridSpecFromSubplotSpec(1, n_models, subplot_spec=gs[1], wspace=0.3)

        import seaborn as sns
        for i, model_name in enumerate(model_names):
            ax_cm = fig.add_subplot(gs_bottom[i])

            model_instance = next(m for m in self.models if m.name == model_name)
            y_pred = model_instance.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False, ax=ax_cm)
            ax_cm.set_title(f"{model_name} Confusion Matrix")
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")

        plt.suptitle('Model Performance Comparison', fontsize=16, y=1.02)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Comparison visuals saved to {output_path}")

