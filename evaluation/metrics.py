# evaluation/metrics.py

# === Import required metrics from scikit-learn ===
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import numpy as np  # For handling NaN values and numerical operations

def compute_classification_metrics(y_true, y_pred, y_proba=None, zero_division=0):
    """
    Calculate standard classification metrics: Accuracy, Precision, Recall, F1 Score, ROC AUC.

    Parameters:
    - y_true: Ground truth (true labels)
    - y_pred: Predicted labels
    - y_proba: Predicted probabilities (for ROC AUC)
    - zero_division: Value to use when zero division occurs in precision/recall

    Returns:
    Dictionary with all computed metric values.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=zero_division)
    recall = recall_score(y_true, y_pred, zero_division=zero_division)
    f1 = f1_score(y_true, y_pred, zero_division=zero_division)

    if y_proba is not None:
        roc_auc = roc_auc_score(y_true, y_proba)
    else:
        roc_auc = np.nan  # ROC AUC not available if probabilities are not given

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    }

def print_classification_metrics(metrics_dict):
    """
    Print formatted output of classification metrics.

    Parameters:
    - metrics_dict: Dictionary of metric names and their values.
    """
    print("\n=== Classification Metrics ===")
    for key, value in metrics_dict.items():
        print(f"{key}: {value:.4f}")

# === Helper function to save and display results ===
import pandas as pd  # For saving and displaying tabular data

def save_and_display_results(results_df, filename, title, show_only_metrics=True):
    """
    Save and optionally filter results for display.

    Parameters:
    - results_df: DataFrame with model results.
    - filename: path to save the CSV file.
    - title: title to print on screen.
    - show_only_metrics: if True, only display and save evaluation metrics.
    """
    # Columns of interest when showing only evaluation metrics
    metric_columns = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "Model", "Search Method"]
    
    if show_only_metrics:
        # Filter only the relevant metric columns that exist in the DataFrame
        display_df = results_df[[col for col in metric_columns if col in results_df.columns]].copy()
    else:
        # Use the full DataFrame
        display_df = results_df.copy()

    # Display section title and formatted metrics
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(display_df.to_string(index=False))

    # Save the filtered or full DataFrame to CSV
    display_df.to_csv(filename, index=False)
    print(f"\nResults saved to: {filename}")
    
    return display_df
