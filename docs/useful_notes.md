## Metrics Description
### Accuracy
- The proportion of all predictions (both positive and negative) that were correct
- Best for balanced datasets where false positives and false negatives are equally important
- Accuracy = (True Positives + True Negatives) / Total Predictions

### Precision
- Of all the patients we predicted would have a stroke, what proportion actually did
- Useful when false positives are costly (e.g., unnecessary medical procedures)
- Precision = True Positives / (True Positives + False Positives)

### Recall (Sensitivity)
- Of all the patients who actually had strokes, what proportion did we correctly identify
- Useful when false negatives are dangerous (missing a stroke prediction could be fatal)
- Recall = True Positives / (True Positives + False Negatives)

### F1 Score
- The harmonic mean of precision and recall, providing a single score that balances both
- Useful when you need to consider both false positives and false negatives
- F1 Score = 2 * (Precision * Recall) / (Precision + Recall)

### ROC AUC (Receiver Operating Characteristic - Area Under the Curve)
- Measures the model's ability to distinguish between classes across all classification thresholds
- Ranges from 0.5 (random guessing) to 1.0 (perfect classification)
- Useful for imbalanced datasets where one class is much more frequent than the other

## Stroke Prediction Specifically
- Recall might be our most important metric because missing a stroke prediction (false negative) could be life-threatening
- The XGBoost model's higher recall (0.38 vs 0.00) makes it potentially more suitable, despite lower precision
- We might want to adjust our classification threshold to increase recall further, even at the cost of precision
- The similar ROC AUC scores suggest both models have comparable discriminative ability, but are using different classification thresholds