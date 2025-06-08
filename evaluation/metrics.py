# evaluation/metrics.py

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import numpy as np

def compute_classification_metrics(y_true, y_pred, y_proba=None, zero_division=0):
    """
    Calcula métricas comunes de clasificación binaria.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=zero_division)
    recall = recall_score(y_true, y_pred, zero_division=zero_division)
    f1 = f1_score(y_true, y_pred, zero_division=zero_division)

    if y_proba is not None:
        roc_auc = roc_auc_score(y_true, y_proba)
    else:
        roc_auc = np.nan  # Si no se proporcionan probabilidades

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    }

def print_classification_metrics(metrics_dict):
    """
    Imprime métricas con formato legible.
    """
    print("\n=== Métricas de Clasificación ===")
    for key, value in metrics_dict.items():
        print(f"{key}: {value:.4f}")
