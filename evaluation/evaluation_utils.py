# evaluation/evaluation_utils.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import pandas as pd

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, model_name=None, save_dir=None):
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
        print(f"Guardado: {path}")
        plt.close()
    else:
        plt.show()

def plot_roc_curve(y_true, y_proba, model_name=None, save_dir=None):
    if y_proba is None:
        print("Este modelo no proporciona probabilidades para la curva ROC.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
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
        print(f"Guardado: {path}")
        plt.close()
    else:
        plt.show()

def plot_feature_importance(model, feature_names, model_name=None, save_dir=None, top_n=20):
    if not hasattr(model, "feature_importances_"):
        print("Este modelo no tiene atributo 'feature_importances_'.")
        return

    importances = model.feature_importances_

    if len(importances) != len(feature_names):
        print("Advertencia: 'importances' y 'feature_names' no coinciden en longitud. Saltando gráfico.")
        return

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
        print(f"Guardado: {path}")
        plt.close()
    else:
        plt.show()

def plot_all_evaluation_graphs(y_true, y_pred, y_proba, feature_importances, feature_names, model_name, save_dir):
    # Rutas específicas para cada tipo de gráfico
    cm_dir = os.path.join(save_dir, "confusion_matrices")
    roc_dir = os.path.join(save_dir, "roc_curves")
    fi_dir = os.path.join(save_dir, "feature_importance")

    plot_confusion_matrix(y_true, y_pred, model_name=model_name, save_dir=cm_dir)
    plot_roc_curve(y_true, y_proba, model_name=model_name, save_dir=roc_dir)

    if feature_importances is not None:
        dummy_model = type("DummyModel", (), {"feature_importances_": feature_importances})
        plot_feature_importance(dummy_model, feature_names, model_name=model_name, save_dir=fi_dir)
