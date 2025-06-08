import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import ParameterGrid

class GridSearchWrapper:
    def __init__(self, model_class, data, param_grid, scoring="f1"):
        """
        model_class: clase del modelo personalizada (por ejemplo, LogisticModel).
        data: DataFrame preprocesado.
        param_grid: diccionario con los parámetros a explorar.
        scoring: métrica de evaluación para seleccionar el mejor modelo.
        """
        self.model_class = model_class
        self.data = data
        self.param_grid = list(ParameterGrid(param_grid))
        self.scoring = scoring
        self.results = []

    def run(self):
        print(f"\n==> Iniciando Grid Search para modelo: {self.model_class.__name__}")

        for i, params in enumerate(self.param_grid):
            print(f"\n→ Grid Search {i+1}/{len(self.param_grid)}: {params}")
            model = self.model_class(self.data, **params)
            model.run_all()
            metrics = model.get_metrics()

            print(f"Métricas obtenidas: {metrics}")

            # Añadir hiperparámetros a las métricas
            metrics_with_params = metrics.copy()
            metrics_with_params.update(params)

            self.results.append(metrics_with_params)

        # Guardar todos los resultados del grid search
        os.makedirs("results/metrics", exist_ok=True)
        results_df = pd.DataFrame(self.results)

        model_name = self.model_class.__name__.replace("Model", "")
        csv_path = f"results/metrics/{model_name}_grid_search_results.csv"
        try:
            results_df.to_csv(csv_path, index=False)
            print(f"Todos los resultados del Grid Search guardados en {csv_path}")
        except Exception as e:
            print(f"Error guardando resultados en {csv_path}: {e}")

        # Seleccionar el mejor modelo según F1 Score y luego Recall
        results_df_sorted = results_df.sort_values(by=["F1 Score", "Recall"], ascending=[False, False])
        best_metrics = results_df_sorted.iloc[0]
        best_params = {k: best_metrics[k] for k in self.param_grid[0].keys() if k in best_metrics}

        # Entrenar el mejor modelo nuevamente
        best_model = self.model_class(self.data, **best_params)
        best_model.run_all()

        # Guardar modelo y métricas en best_models/
        os.makedirs("results/best_models", exist_ok=True)
        model_class_name = best_model.model.__class__.__name__

        best_model_path = os.path.join("results/best_models", f"{model_class_name}.pkl")
        best_metrics_path = os.path.join("results/best_models", f"{model_class_name}_metrics.csv")

        try:
            joblib.dump(best_model.model, best_model_path)
            print(f"Mejor modelo de tipo {model_class_name} guardado en {best_model_path}")
        except Exception as e:
            print(f"Error guardando modelo {model_class_name}: {e}")

        try:
            pd.DataFrame([best_metrics]).to_csv(best_metrics_path, index=False)
            print(f"Métricas del mejor modelo guardadas en {best_metrics_path}")
        except Exception as e:
            print(f"Error guardando métricas del modelo {model_class_name}: {e}")

        print("\n=== Mejor configuración encontrada ===")
        print(f"Parámetros: {best_params}")
        print(f"F1 Score: {best_metrics['F1 Score']:.4f} | Recall: {best_metrics['Recall']:.4f}")

        return best_model, best_metrics["F1 Score"], best_metrics.to_dict()

    def _normalize_metric_name(self):
        """Asegura que la métrica exista en los resultados del modelo."""
        valid_metrics = ["Accuracy", "F1 Score", "ROC AUC", "Precision", "Recall"]
        for metric in valid_metrics:
            if self.scoring.lower() in metric.lower():
                return metric
        raise ValueError(f"Métrica '{self.scoring}' no es válida. Usa una de: {valid_metrics}")
