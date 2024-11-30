from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ModelComparison:
    def __init__(self, models, X, y):
        self.models = models
        logger.info(f"Initializing model comparison with {len(models)} models")

        logger.debug("Splitting data into train and test sets")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(f"Train set size: {self.X_train.shape}, Test set size: {self.X_test.shape}")
        self.results = {}

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