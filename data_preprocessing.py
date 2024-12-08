# data_preprocessing.py

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder

logger = logging.getLogger(__name__)

# Configuration parameters
NUMERIC_COLUMNS = ['age', 'avg_glucose_level', 'bmi']
CATEGORICAL_COLUMNS = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
BINARY_COLUMNS = ['hypertension', 'heart_disease', 'stroke']
INTERACTION_FEATURES = True
CORRELATION_MATRIX_GENERATION = False
QUANTILE_RANGE = (25.0, 75.0)
STRONG_CORR_THRESHOLD = 0.5


def get_age_group(age):
    """Classify age into groups for BMI imputation."""
    if age <= 20:
        return '0-20'
    elif age <= 40:
        return '21-40'
    elif age <= 60:
        return '41-60'
    else:
        return '61-80'


def impute_bmi_by_age_group(df):
    """Impute missing BMI values based on age group means."""
    logger.info("Imputing BMI values by age groups")
    df['age_group'] = df['age'].apply(get_age_group)

    # Compute mean BMI for each group and impute
    age_group_means = df.groupby('age_group')['bmi'].mean()
    logger.debug(f"Age group BMI means:\n{age_group_means}")

    for age_group, mean_bmi in age_group_means.items():
        mask = (df['age_group'] == age_group) & (df['bmi'].isna())
        df.loc[mask, 'bmi'] = mean_bmi

    # Fallback to overall mean if still missing
    if df['bmi'].isna().any():
        overall_mean = df['bmi'].mean()
        df['bmi'].fillna(overall_mean, inplace=True)
        logger.warning(f"Used overall mean ({overall_mean:.2f}) for remaining missing BMI values")

    df.drop('age_group', axis=1, inplace=True)
    return df


def encode_categorical_features(df):
    """One-hot encode categorical features."""
    logger.info("One-hot encoding categorical variables")
    onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = onehot.fit_transform(df[CATEGORICAL_COLUMNS])
    feature_names = onehot.get_feature_names_out(CATEGORICAL_COLUMNS)

    encoded_df = pd.DataFrame(encoded_cats, columns=feature_names, index=df.index)
    df = pd.concat([df.drop(CATEGORICAL_COLUMNS, axis=1), encoded_df], axis=1)

    logger.debug(f"Shape after one-hot encoding: {df.shape}")
    logger.debug(f"Features after encoding: {list(df.columns)}")
    return df, onehot


def handle_outliers_and_scale(df, numeric_columns):
    """
    Handle outliers using RobustScaler and then normalize using StandardScaler.
    """
    logger.info("Handling outliers and scaling numeric features")
    try:
        original_stats = df[numeric_columns].describe()

        # Robust scaling
        robust_scaler = RobustScaler(quantile_range=QUANTILE_RANGE)
        robust_scaled = robust_scaler.fit_transform(df[numeric_columns])

        # Standard scaling
        standard_scaler = StandardScaler()
        final_scaled = standard_scaler.fit_transform(robust_scaled)

        df[numeric_columns] = final_scaled
        scaled_stats = df[numeric_columns].describe()

        # Log before/after stats
        for column in numeric_columns:
            logger.info(f"\nFeature: {column}")
            logger.info("Before scaling:")
            logger.info(f"Mean: {original_stats.loc['mean', column]:.3f}, Std: {original_stats.loc['std', column]:.3f}, "
                        f"Min: {original_stats.loc['min', column]:.3f}, Max: {original_stats.loc['max', column]:.3f}")
            logger.info("After scaling:")
            logger.info(f"Mean: {scaled_stats.loc['mean', column]:.3f}, Std: {scaled_stats.loc['std', column]:.3f}, "
                        f"Min: {scaled_stats.loc['min', column]:.3f}, Max: {scaled_stats.loc['max', column]:.3f}")

        return df, robust_scaler, standard_scaler

    except Exception as e:
        logger.error(f"Error in outlier handling and scaling: {str(e)}", exc_info=True)
        raise


def create_interaction_features(df):
    """Create interaction features to improve model representation."""
    logger.info("Creating interaction features")
    df['age_hypertension'] = df['age'] * df['hypertension']
    df['age_heart_disease'] = df['age'] * df['heart_disease']
    df['bmi_glucose'] = df['bmi'] * df['avg_glucose_level']
    df['age_bmi'] = df['age'] * df['bmi']
    df['hypertension_heart_disease'] = df['hypertension'] * df['heart_disease']
    return df


def generate_correlation_matrix(df, output_path='correlation_matrix.png'):
    """Generate and save a correlation matrix visualization."""
    logger.info("Generating correlation matrix")
    try:
        corr_matrix = df.corr()

        # Plot correlation matrix
        plt.figure(figsize=(20, 16))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f',
            square=True,
            mask=mask
        )
        plt.title('Feature Correlation Matrix', pad=20)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Correlation matrix saved to {output_path}")

        # Identify strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) >= STRONG_CORR_THRESHOLD:
                    strong_correlations.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.index[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })

        strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        if strong_correlations:
            logger.info("Strong feature correlations found (|corr| >= 0.5):")
            for corr in strong_correlations:
                logger.info(f"{corr['feature1']} - {corr['feature2']}: {corr['correlation']:.3f}")

        return corr_matrix, strong_correlations

    except Exception as e:
        logger.error(f"Error generating correlation matrix: {str(e)}", exc_info=True)
        raise


def preprocess_data(df):
    """Full preprocessing pipeline for the stroke dataset."""
    logger.info("Starting data preprocessing")
    df = df.copy()

    # Drop ID column if it exists
    if 'id' in df.columns:
        logger.debug("Dropping 'id' column")
        df.drop('id', axis=1, inplace=True, errors='ignore')

    # Impute missing values
    df = impute_bmi_by_age_group(df)

    # Encode categorical features
    df, onehot = encode_categorical_features(df)

    # Ensure correct data types (float32 for numeric)
    for column in df.columns:
        df[column] = df[column].astype(np.float32)

    # Handle outliers and scale numeric features
    df, robust_scaler, standard_scaler = handle_outliers_and_scale(df, NUMERIC_COLUMNS)

    # Optionally create interaction features
    if INTERACTION_FEATURES:
        df = create_interaction_features(df)

    # Generate correlation matrix (optional - can disable if not needed every run)
    if CORRELATION_MATRIX_GENERATION:
        generate_correlation_matrix(df)

    # Separate features and target
    if 'stroke' not in df.columns:
        raise ValueError("Target column 'stroke' not found in the dataset.")

    y = df['stroke'].astype(np.float32)
    df.drop('stroke', axis=1, inplace=True)

    # Example of selecting final features (modify as needed)
    final_features = [
        'age',
        'avg_glucose_level',
        'bmi',
        'hypertension',
        'heart_disease',
        'ever_married_No',
        'ever_married_Yes',
        'age_hypertension',
        'age_heart_disease'
    ]

    X = df[final_features]

    logger.debug(f"Final features: {X.columns.tolist()}")
    logger.info(f"Class distribution:\n{y.value_counts(normalize=True)}")

    return X, y, (robust_scaler, standard_scaler, onehot)


def load_and_preprocess(file_path):
    """Load the dataset from CSV and run preprocessing pipeline."""
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Initial data shape: {df.shape}")
        logger.debug(f"Initial class distribution:\n{df['stroke'].value_counts(normalize=True)}")

        X, y, scalers = preprocess_data(df)
        logger.info(f"Final data shape: {X.shape}")
        return X, y

    except Exception as e:
        logger.error(f"Error loading or preprocessing data: {str(e)}", exc_info=True)
        raise
