import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
data_path = "data/healthcare-dataset-stroke-data.csv"
df = pd.read_csv(data_path)

# Data preprocessing
def preprocess_data(df):
    # Drop irrelevant column
    df = df.drop(['id'], axis=1)

    # Handle missing values (e.g., BMI)
    imputer = SimpleImputer(strategy='median')
    df['bmi'] = imputer.fit_transform(df[['bmi']])

    # Encode categorical variables
    categorical_columns = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Feature scaling for numerical features
    scaler = StandardScaler()
    numerical_features = ['age', 'avg_glucose_level', 'bmi']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Split features and target
    X = df.drop('stroke', axis=1)
    y = df['stroke']

    return X, y

X, y = preprocess_data(df)

# Address class imbalance using oversampling (SMOTE)
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# KNN model
def train_knn(X_train, y_train, X_test, y_test):
    # Grid search for hyperparameter tuning
    param_grid = {
        'n_neighbors': range(1, 21),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='roc_auc', verbose=1)
    grid_search.fit(X_train, y_train)

    # Best model
    best_knn = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")

    # Predictions
    y_pred = best_knn.predict(X_test)
    y_prob = best_knn.predict_proba(X_test)[:, 1]

    # Metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

    return best_knn

# Train and evaluate the KNN model
knn_model = train_knn(X_train, y_train, X_test, y_test)