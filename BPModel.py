import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split

# load and preprocess
data = pd.read_csv('healthcare-dataset-stroke-data.csv')

data = data.drop(columns=["id"])

data["bmi"].fillna(data["bmi"].median(), inplace=True)

categorical_features = ["gender", "ever_married", "work_type", "Residence_type",
                        "smoking_status"]
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)
X = data.drop(columns=["stroke"])
y = data["stroke"]

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42,
                                                    stratify=y_resampled)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
classification_rep = classification_report(y_test, y_pred)
confusion_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy: ", accuracy)
print("ROC AUC: ", roc_auc)
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", confusion_matrix)

sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["No Stroke", "Stroke"],
            yticklabels=["No Stroke", "Stroke"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("ROC Curve")
plt.show()
