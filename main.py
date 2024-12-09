from data_preprocessing import load_and_preprocess
from logging_config import setup_logging
from model_comparison import ModelComparison
from neural_network import StrokeNN
from knn_model import StrokeKNN
from xgboost_model import StrokeXGBoost

if __name__ == "__main__":
    # Set up logging
    logger = setup_logging()
    logger.info("Starting stroke prediction model training")

    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data")
        X, y = load_and_preprocess("data/healthcare-dataset-stroke-data.csv")

        # Initialize models
        logger.info("Initializing models")
        nn_model = StrokeNN(input_size=X.shape[1])
        xgb_model = StrokeXGBoost()
        knn_model = StrokeKNN()

        # Compare models
        logger.info("Starting model comparison")
        comparison = ModelComparison([knn_model, xgb_model], X, y)
        comparison.train_and_evaluate()
        comparison.print_results()

        logger.info("Model training and evaluation completed successfully")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise
