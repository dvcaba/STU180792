# evaluation/evaluation_utils.py

import os  # For handling directories
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For enhanced visualization (heatmaps, bar plots)
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score  # Evaluation metrics
import pandas as pd  # For data handling (used in feature importance plot)
import shutil  # For file operations (used in renaming and moving plots)

def ensure_dir(path):
    """Create the directory if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, model_name=None, save_dir=None):
    """
    Plot a confusion matrix using Seaborn heatmap.
    
    If model_name and save_dir are provided, saves the plot to disk;
    otherwise, displays it interactively.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    if save_dir and model_name:
        ensure_dir(save_dir)
        path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(path)
        print(f"Saved: {path}")
        plt.close()
    else:
        plt.show()

def plot_roc_curve(y_true, y_proba, model_name=None, save_dir=None):
    """
    Plot ROC curve using true labels and predicted probabilities.
    
    Saves or shows the curve depending on parameters.
    """
    if y_proba is None:
        print("This model does not provide probabilities for the ROC curve.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_proba)  # Get false/true positive rates
    auc_score = roc_auc_score(y_true, y_proba)  # Compute AUC

    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal reference line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_dir and model_name:
        ensure_dir(save_dir)
        path = os.path.join(save_dir, f"{model_name}_roc_curve.png")
        plt.savefig(path)
        print(f"Saved: {path}")
        plt.close()
    else:
        plt.show()

def plot_feature_importance(model, feature_names, model_name=None, save_dir=None, top_n=20):
    """
    Plot feature importance for models that expose `feature_importances_` attribute.
    
    Plots top `top_n` features. Saves or shows plot based on parameters.
    """
    if not hasattr(model, "feature_importances_"):
        print("This model lacks 'feature_importances_' attribute.")
        return

    importances = model.feature_importances_

    if len(importances) != len(feature_names):
        print("Warning: 'importances' and 'feature_names' length mismatch. Skipping plot.")
        return

    # Create DataFrame for visualization
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x="Importance", y="Feature")
    plt.title("Feature Importances")
    plt.grid(True)
    plt.tight_layout()

    if save_dir and model_name:
        ensure_dir(save_dir)
        path = os.path.join(save_dir, f"{model_name}_feature_importance.png")
        plt.savefig(path)
        print(f"Saved: {path}")
        plt.close()
    else:
        plt.show()

def plot_all_evaluation_graphs(y_true, y_pred, y_proba, feature_importances, feature_names, model_name, save_dir):
    """
    Convenience function to plot all evaluation metrics and save them in organized folders.
    
    Generates:
    - Confusion matrix
    - ROC curve
    - Feature importance (if available)
    """
    # Create separate subdirectories for each type of plot
    cm_dir = os.path.join(save_dir, "confusion_matrices")
    roc_dir = os.path.join(save_dir, "roc_curves")
    fi_dir = os.path.join(save_dir, "feature_importance")

    # Generate and save each plot
    plot_confusion_matrix(y_true, y_pred, model_name=model_name, save_dir=cm_dir)
    plot_roc_curve(y_true, y_proba, model_name=model_name, save_dir=roc_dir)

    if feature_importances is not None:
        # Create a dummy object with the required attribute for plotting
        dummy_model = type("DummyModel", (), {"feature_importances_": feature_importances})
        plot_feature_importance(dummy_model, feature_names, model_name=model_name, save_dir=fi_dir)

# === Helper to rename plots with contextual info ===
def rename_model_plots(model_name, source_dirs, dest_dir, prefix, execution_id=None):
    """
    Rename and copy evaluation plots with informative filenames.

    Parameters:
    - model_name: the name of the model used
    - source_dirs: list of source directories to look for plots
    - dest_dir: destination folder to move renamed plots
    - prefix: experiment or project prefix for naming
    - execution_id: optional suffix to identify specific execution runs
    """
    suffix = f"_execution_{execution_id}" if execution_id else ""

    # Define mapping from original file to renamed version
    plot_mappings = {
        f"{model_name}_confusion_matrix.png": f"{prefix}_{model_name}_confusion_matrix{suffix}.png",
        f"{model_name}_roc_curve.png": f"{prefix}_{model_name}_roc_curve{suffix}.png",
        f"{model_name}_feature_importance.png": f"{prefix}_{model_name}_feature_importance{suffix}.png"
    }

    # Search and copy files from each directory in source_dirs
    for source_dir in source_dirs:
        for old_name, new_name in plot_mappings.items():
            src = os.path.join(source_dir, old_name.replace(f"{prefix}_", ""))
            dst = os.path.join(dest_dir, new_name)
            if os.path.exists(src):
                shutil.copy2(src, dst)
