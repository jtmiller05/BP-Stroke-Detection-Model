# data_preprocessing.py

import logging

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


def preprocess_data(df):
    """
    Preprocess the stroke prediction dataset
    """
    logger.info("Starting data preprocessing")

    # Create a copy to avoid modifying the original dataframe
    df = df.copy()

    # 1. Drop the 'id' column as it's just an identifier
    logger.debug("Dropping 'id' column")
    df = df.drop('id', axis=1)

    # 2. Handle missing values
    logger.info("Handling missing values")
    missing_values = df.isnull().sum()
    logger.debug(f"Missing values before imputation:\n{missing_values}")

    # For numeric columns (bmi), impute with median
    numeric_imputer = SimpleImputer(strategy='median')
    df['bmi'] = numeric_imputer.fit_transform(df[['bmi']])
    logger.info("Completed BMI imputation")

    # 3. Handle categorical variables
    logger.info("Processing categorical variables")

    # Initialize dictionary to store label encoders
    label_encoders = {}

    # Encode all categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column].astype(str))
        logger.debug(f"Encoded {column} with {len(label_encoders[column].classes_)} unique values")

    # 4. Convert all columns to float32 for better compatibility with PyTorch
    for column in df.columns:
        if column != 'stroke':  # Keep target variable as is
            df[column] = df[column].astype(np.float32)

    # 5. Feature scaling
    logger.info("Scaling numeric features")
    numeric_features = ['age', 'avg_glucose_level', 'bmi']
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # 6. Create interaction features
    logger.info("Creating interaction features")
    df['age_hypertension'] = df['age'] * df['hypertension']
    df['age_heart_disease'] = df['age'] * df['heart_disease']
    df['bmi_glucose'] = df['bmi'] * df['avg_glucose_level']
    df['age_bmi'] = df['age'] * df['bmi']  # New interaction feature
    df['hypertension_heart_disease'] = df['hypertension'] * df['heart_disease']  # New interaction feature

    # Ensure all feature columns are float32
    feature_columns = [col for col in df.columns if col != 'stroke']
    df[feature_columns] = df[feature_columns].astype(np.float32)

    # 7. Separate features and target
    X = df.drop('stroke', axis=1)
    y = df['stroke'].astype(np.float32)  # Convert target to float32 for PyTorch

    logger.debug(f"Final features: {X.columns.tolist()}")
    logger.info(f"Class distribution:\n{y.value_counts(normalize=True)}")

    # Verify data types
    logger.debug(f"Feature dtypes:\n{X.dtypes}")
    logger.debug(f"Target dtype: {y.dtype}")

    return X, y


def load_and_preprocess(file_path):
    """
    Load and preprocess the dataset
    """
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Initial data shape: {df.shape}")
        logger.debug(f"Initial class distribution:\n{df['stroke'].value_counts(normalize=True)}")

        X, y = preprocess_data(df)
        logger.info(f"Final data shape: {X.shape}")
        return X, y

    except Exception as e:
        logger.error(f"Error loading or preprocessing data: {str(e)}", exc_info=True)
        raise
