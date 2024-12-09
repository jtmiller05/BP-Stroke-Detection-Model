# neural_network.py

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model_base import StrokeModel

logger = logging.getLogger(__name__)


def get_device():
    """
    Get the best available device for PyTorch.
    Returns device and a printable status message.
    """
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            status = f"Using GPU: {device_name} ({memory_gb:.1f}GB)"
        else:
            device = torch.device('cpu')
            status = "Using CPU (GPU not available)"

        logger.info(status)
        return device, status

    except Exception as e:
        logger.error(f"Error determining device: {str(e)}", exc_info=True)
        raise


class NeuralNetworkModule(nn.Module):
    """PyTorch neural network module"""

    def __init__(self, input_size, hidden_sizes=[128, 64, 32]):
        super().__init__()
        logger.info("Initializing Neural Network architecture")
        logger.debug(f"Input size: {input_size}, Hidden sizes: {hidden_sizes}")
        self.layers = self._build_layers(input_size, hidden_sizes)

    def _build_layers(self, input_size, hidden_sizes):
        sizes = [input_size] + hidden_sizes + [1]
        layers = []

        try:
            for i in range(len(sizes) - 1):
                logger.debug(f"Building layer {i + 1}: {sizes[i]} -> {sizes[i + 1]}")
                layers.append(nn.Linear(sizes[i], sizes[i + 1]))

                if i < len(sizes) - 2:
                    layers.append(nn.ReLU())
                    layers.append(nn.BatchNorm1d(sizes[i + 1]))
                    layers.append(nn.Dropout(0.5))  # Increased dropout to reduce overfitting
                else:
                    layers.append(nn.Identity())  # Remove activation for output layer

            return nn.Sequential(*layers)

        except Exception as e:
            logger.error(f"Error building network layers: {str(e)}", exc_info=True)
            raise

    def forward(self, x):
        return self.layers(x).squeeze()


class StrokeNN(StrokeModel):
    """Neural Network model for stroke prediction that implements the StrokeModel interface"""

    def __init__(self, input_size, hidden_sizes=[256, 128, 64, 32], learning_rate=0.0001):
        super().__init__("Neural Network")
        logger.info("Initializing StrokeNN model")

        try:
            self.device, device_status = get_device()

            # Initialize the neural network
            self.network = NeuralNetworkModule(input_size, hidden_sizes)
            self.network.to(self.device)

            # Placeholder for criterion, will be initialized during training
            self.criterion = None
            self.optimizer = torch.optim.Adam(
                self.network.parameters(), lr=learning_rate, weight_decay=1e-5
            )
            self.best_model_path = 'best_nn_model.pth'

            logger.debug(f"Learning rate: {learning_rate}")
            logger.debug(f"Model architecture:\n{self.network}")

        except Exception as e:
            logger.error(f"Error initializing StrokeNN: {str(e)}", exc_info=True)
            raise

    def _prepare_data(self, X, y=None):
        """Convert data to PyTorch tensors and move to device"""
        logger.debug("Preparing data for PyTorch")
        try:
            X = torch.FloatTensor(X.values).to(self.device)
            if y is not None:
                y = torch.FloatTensor(y.values).to(self.device)
                return X, y
            return X
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}", exc_info=True)
            raise

    def tune_hyperparameters(self, X_train, y_train, X_val, y_val, n_trials=50):
        """Tune hyperparameters using Optuna"""
        raise NotImplementedError("Hyperparameter tuning is not implemented for Neural Networks")

    def train(self, X, y, batch_size=128, epochs=1000, validation_split=0.2):
        """Train the neural network"""
        logger.info("Starting neural network training")

        try:
            # Prepare data
            X, y = self._prepare_data(X, y)
            val_size = int(len(X) * validation_split)
            indices = torch.randperm(len(X))

            train_indices = indices[val_size:]
            val_indices = indices[:val_size]

            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]

            # Calculate positive class weight
            pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
            pos_weight = pos_weight.to(self.device)

            # Initialize criterion with pos_weight
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            best_val_loss = float('inf')
            patience = 20
            patience_counter = 0

            logger.info(f"Starting training for {epochs} epochs")
            for epoch in range(epochs):
                # Training phase
                self.network.train()
                train_loss = 0

                for batch_X, batch_y in train_loader:
                    self.optimizer.zero_grad()
                    outputs = self.network(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()

                # Validation phase
                self.network.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.network(batch_X)
                        val_loss += self.criterion(outputs, batch_y).item()

                # Calculate average losses
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)

                logger.info(
                    f"Epoch [{epoch + 1}/{epochs}] - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
                )

                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(self.network.state_dict(), self.best_model_path)
                    logger.debug(f"Saved new best model with validation loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        self.network.load_state_dict(torch.load(self.best_model_path))
                        break

            logger.info("Training completed")

        except Exception as e:
            logger.error(f"Error during training: {str(e)}", exc_info=True)
            raise

    def predict(self, X):
        """Predict class labels"""
        logger.debug(f"Making predictions for {len(X)} samples")
        try:
            self.network.eval()
            X = self._prepare_data(X)

            with torch.no_grad():
                outputs = self.network(X)
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).cpu().numpy()

            logger.debug(f"Prediction distribution: {np.bincount(predictions.astype(int))}")
            return predictions

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            raise

    def predict_proba(self, X):
        """Predict class probabilities"""
        logger.debug(f"Computing prediction probabilities for {len(X)} samples")
        try:
            self.network.eval()
            X = self._prepare_data(X)

            with torch.no_grad():
                outputs = self.network(X)
                probabilities = torch.sigmoid(outputs).cpu().numpy()

            logger.debug(
                f"Probability statistics - Mean: {np.mean(probabilities):.4f}, "
                f"Std: {np.std(probabilities):.4f}"
            )
            return probabilities

        except Exception as e:
            logger.error(f"Error during probability prediction: {str(e)}", exc_info=True)
            raise
