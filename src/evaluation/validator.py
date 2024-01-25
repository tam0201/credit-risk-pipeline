import torch
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import numpy as np

def validate_nn(model, data_loader):
    """Validate a neural network model using a DataLoader."""
    model.eval()  # Set the model to evaluation mode
    predictions = []
    probabilities = []
    true_labels = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch)
            # Assuming binary classification with a single output neuron
            predicted_probs = outputs.cpu().numpy()
            predicted_labels = (predicted_probs > 0.5).astype(np.int32)
            predictions.extend(predicted_labels)
            probabilities.extend(predicted_probs)
            true_labels.extend(y_batch.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    roc_auc = roc_auc_score(true_labels, probabilities)
    classification_rep = classification_report(true_labels, predictions, target_names=['Class 0', 'Class 1'])

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("Classification Report:")
    print(classification_rep)
    
    return accuracy, roc_auc, classification_rep

def validate_gbm(model, X_test, y_test):
    """Validate a gradient boosting model using the full test dataset."""
    # Gradient boosting models typically provide predict_proba for classification
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities > 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)
    classification_rep = classification_report(y_test, predictions, target_names=['Class 0', 'Class 1'])

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("Classification Report:")
    print(classification_rep)

    return accuracy, roc_auc, classification_rep