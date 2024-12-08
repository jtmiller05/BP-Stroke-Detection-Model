# data_preprocessing.py

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder

logger = logging.getLogger(__name__)


def handle_outliers_and_scale(df, numeric_columns):
    """
    Handle outliers using RobustScaler and then normalize using StandardScaler

    Args:
        df (pd.DataFrame): Input dataframe
        numeric_columns (list): List of numeric column names
    """
    logger.info("Handling outliers and scaling numeric features")

    try:
        # Create copies of the original numeric data for comparison
        original_stats = df[numeric_columns].describe()

        # First apply RobustScaler to handle outliers
        robust_scaler = RobustScaler(quantile_range=(25.0, 75.0))
        robust_scaled = robust_scaler.fit_transform(df[numeric_columns])

        # Then apply StandardScaler for consistent feature ranges
        standard_scaler = StandardScaler()
        final_scaled = standard_scaler.fit_transform(robust_scaled)

        # Update the dataframe with scaled values
        df[numeric_columns] = final_scaled

        # Calculate and log statistics before and after scaling
        scaled_stats = df[numeric_columns].describe()

        logger.info("Feature statistics before and after scaling:")
        for column in numeric_columns:
            logger.info(f"\nFeature: {column}")
            logger.info("Before scaling:")
            logger.info(f"Mean: {original_stats.loc['mean', column]:.3f}")
            logger.info(f"Std: {original_stats.loc['std', column]:.3f}")
            logger.info(f"Min: {original_stats.loc['min', column]:.3f}")
            logger.info(f"Max: {original_stats.loc['max', column]:.3f}")

            logger.info("After scaling:")
            logger.info(f"Mean: {scaled_stats.loc['mean', column]:.3f}")
            logger.info(f"Std: {scaled_stats.loc['std', column]:.3f}")
            logger.info(f"Min: {scaled_stats.loc['min', column]:.3f}")
            logger.info(f"Max: {scaled_stats.loc['max', column]:.3f}")

        return df, robust_scaler, standard_scaler

    except Exception as e:
        logger.error(f"Error in outlier handling and scaling: {str(e)}", exc_info=True)
        raise

def generate_correlation_matrix(df, output_path='correlation_matrix.png'):
    """
    Generate and save a correlation matrix visualization

    Args:
        df (pd.DataFrame): The preprocessed feature dataframe
        output_path (str): Path to save the correlation matrix plot
    """
    logger.info("Generating correlation matrix")

    try:
        # Calculate correlation matrix
        corr_matrix = df.corr()

        # Create figure with appropriate size
        plt.figure(figsize=(20, 16))

        # Create heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,  # Show correlation values
            cmap='coolwarm',  # Blue to red colormap
            center=0,  # Center the colormap at 0
            fmt='.2f',  # Show 2 decimal places
            square=True,  # Make cells square
            mask=np.triu(np.ones_like(corr_matrix, dtype=bool))  # Show only lower triangle
        )

        plt.title('Feature Correlation Matrix', pad=20)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Save the plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Correlation matrix saved to {output_path}")

        # Find and log strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) >= 0.5:  # Threshold for strong correlation
                    strong_correlations.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.index[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })

        # Sort by absolute correlation value
        strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

        # Log strong correlations
        if strong_correlations:
            logger.info("Strong feature correlations found (|correlation| >= 0.5):")
            for corr in strong_correlations:
                logger.info(f"{corr['feature1']} - {corr['feature2']}: {corr['correlation']:.3f}")

        return corr_matrix, strong_correlations

    except Exception as e:
        logger.error(f"Error generating correlation matrix: {str(e)}", exc_info=True)
        raise


def get_age_group(age):
    """
    Classify age into groups
    """
    if age <= 20:
        return '0-20'
    elif age <= 40:
        return '21-40'
    elif age <= 60:
        return '41-60'
    else:
        return '61-80'


def impute_bmi_by_age_group(df):
    """
    Impute missing BMI values based on age group means
    """
    logger.info("Imputing BMI values based on age groups")

    # Create age groups
    df['age_group'] = df['age'].apply(get_age_group)

    # Calculate mean BMI for each age group
    age_group_means = df.groupby('age_group')['bmi'].mean()
    logger.debug(f"Age group BMI means:\n{age_group_means}")

    # Impute missing values
    for age_group in age_group_means.index:
        mask = (df['age_group'] == age_group) & (df['bmi'].isna())
        df.loc[mask, 'bmi'] = age_group_means[age_group]

    # If any BMI values are still missing (e.g., in case of empty age groups),
    # use the overall mean as a fallback
    if df['bmi'].isna().any():
        overall_mean = df['bmi'].mean()
        df['bmi'].fillna(overall_mean, inplace=True)
        logger.warning(f"Used overall mean ({overall_mean:.2f}) for remaining missing BMI values")

    # Drop the temporary age_group column
    df.drop('age_group', axis=1, inplace=True)

    return df


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

    # Impute BMI values using age group means
    df = impute_bmi_by_age_group(df)
    logger.info("Completed BMI imputation by age groups")

    # 3. Handle categorical variables with one-hot encoding
    logger.info("Processing categorical variables with one-hot encoding")

    # Initialize dictionary to store label encoders
    categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    binary_columns = ['hypertension', 'heart_disease', 'stroke']  # These are already binary
    numeric_columns = ['age', 'avg_glucose_level', 'bmi']

    # Encode all categorical columns
    onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Fit and transform categorical columns
    encoded_cats = onehot.fit_transform(df[categorical_columns])

    # Get feature names after encoding
    feature_names = onehot.get_feature_names_out(categorical_columns)

    # Create a new dataframe with encoded variables
    encoded_df = pd.DataFrame(
        encoded_cats,
        columns=feature_names,
        index=df.index
    )

    # Combine with numeric and binary columns
    final_df = pd.concat([
        df[numeric_columns + binary_columns],
        encoded_df
    ], axis=1)

    logger.debug(f"Shape after one-hot encoding: {final_df.shape}")
    logger.debug(f"New features after encoding: {final_df.columns.tolist()}")

    # 4. Convert all columns to float32 for better compatibility with PyTorch
    for column in final_df.columns:
        final_df[column] = final_df[column].astype(np.float32)

    # 5. Handle outliers and scale numeric features
    final_df, robust_scaler, standard_scaler = handle_outliers_and_scale(final_df, numeric_columns)

    # 6. Create interaction features
    logger.info("Creating interaction features")
    final_df['age_hypertension'] = final_df['age'] * final_df['hypertension']
    final_df['age_heart_disease'] = final_df['age'] * final_df['heart_disease']
    final_df['bmi_glucose'] = final_df['bmi'] * final_df['avg_glucose_level']
    final_df['age_bmi'] = final_df['age'] * final_df['bmi']
    final_df['hypertension_heart_disease'] = final_df['hypertension'] * final_df['heart_disease']

    generate_correlation_matrix(final_df)

    final_features = [
        'age',
        'avg_glucose_level',
        'bmi',
        'hypertension',
        'heart_disease',
        'ever_married_No',
        'ever_married_Yes',
        # 'smoking_status_Unknown',
        # 'smoking_status_formerly smoked',
        # 'smoking_status_never smoked',
        # 'smoking_status_smokes',
        'age_hypertension',
        'age_heart_disease',
        # 'bmi_glucose',
        # 'age_bmi',
        # 'hypertension_heart_disease',
    ]

    # 7. Separate features and target
    X = final_df.drop('stroke', axis=1)[final_features]
    y = final_df['stroke'].astype(np.float32)  # Convert target to float32 for PyTorch

    logger.debug(f"Final features: {X.columns.tolist()}")
    logger.info(f"Class distribution:\n{y.value_counts(normalize=True)}")

    # Verify data types
    logger.debug(f"Feature dtypes:\n{X.dtypes}")
    logger.debug(f"Target dtype: {y.dtype}")

    return X, y, (robust_scaler, standard_scaler)


def load_and_preprocess(file_path):
    """
    Load and preprocess the dataset
    """
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Initial data shape: {df.shape}")
        logger.debug(f"Initial class distribution:\n{df['stroke'].value_counts(normalize=True)}")

        X, y, (robust_scaler, standard_scaler) = preprocess_data(df)
        logger.info(f"Final data shape: {X.shape}")
        return X, y

    except Exception as e:
        logger.error(f"Error loading or preprocessing data: {str(e)}", exc_info=True)
        raise
