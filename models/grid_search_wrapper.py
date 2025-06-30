import os  # For creating directories and handling file paths
import pandas as pd  # For handling data frames and saving results
import joblib  # For saving trained models to disk
from sklearn.model_selection import ParameterGrid  # For grid search parameter combinations
import io  # For redirecting stdout to a buffer
from contextlib import redirect_stdout  # Used to capture print statements during grid search

class GridSearchWrapper:
    def __init__(self, model_class, data, param_grid, scoring="f1"):
        """
        Initialize the grid search wrapper.

        Parameters:
        - model_class: custom model class to be trained and evaluated
        - data: dataset to be passed to the model
        - param_grid: dictionary of hyperparameters for grid search
        - scoring: metric used to select the best model (default: F1)
        """
        self.model_class = model_class
        self.data = data
        self.param_grid = list(ParameterGrid(param_grid))  # Create all parameter combinations
        self.scoring = scoring
        self.results = []  # Store metrics for all configurations

    def run(self):
        """
        Perform grid search over all parameter combinations.

        Trains a model for each configuration, evaluates it,
        saves results, selects the best model, and stores it.
        """
        print(f"\n==> Starting Grid Search for model: {self.model_class.__name__}")

        for i, params in enumerate(self.param_grid):
            print(f"\n→ Grid Search {i+1}/{len(self.param_grid)}: {params}")
            model = self.model_class(self.data, **params)  # Instantiate model with parameters
            model.run_all()  # Train and evaluate the model
            metrics = model.get_metrics()  # Get evaluation metrics

            print(f"Metrics obtained: {metrics}")

            # Add hyperparameters to metrics for saving
            metrics_with_params = metrics.copy()
            metrics_with_params.update(params)

            self.results.append(metrics_with_params)

        # Save all grid search results to CSV
        os.makedirs("results/metrics", exist_ok=True)
        results_df = pd.DataFrame(self.results)

        model_name = self.model_class.__name__.replace("Model", "")
        csv_path = f"results/metrics/{model_name}_grid_search_results.csv"
        try:
            results_df.to_csv(csv_path, index=False)
            print(f"All grid search results saved to {csv_path}")
        except Exception as e:
            print(f"Error saving results to {csv_path}: {e}")

        # Select best model by F1 Score, then Recall
        results_df_sorted = results_df.sort_values(by=["F1 Score", "Recall"], ascending=[False, False])
        best_metrics = results_df_sorted.iloc[0]
        best_params = {k: best_metrics[k] for k in self.param_grid[0].keys() if k in best_metrics}

        # Train best model again with optimal parameters
        best_model = self.model_class(self.data, **best_params)
        best_model.run_all()

        # Save best model and its metrics
        os.makedirs("results/best_models", exist_ok=True)
        model_class_name = best_model.model.__class__.__name__

        best_model_path = os.path.join("results/best_models", f"{model_class_name}.pkl")
        best_metrics_path = os.path.join("results/best_models", f"{model_class_name}_metrics.csv")

        try:
            joblib.dump(best_model.model, best_model_path)  # Save model to disk
            print(f"Best model of type {model_class_name} saved to {best_model_path}")
        except Exception as e:
            print(f"Error saving model {model_class_name}: {e}")

        try:
            pd.DataFrame([best_metrics]).to_csv(best_metrics_path, index=False)  # Save best metrics
            print(f"Best model metrics saved to {best_metrics_path}")
        except Exception as e:
            print(f"Error saving metrics for model {model_class_name}: {e}")

        print("\n=== Best configuration found ===")
        print(f"Parameters: {best_params}")
        print(f"F1 Score: {best_metrics['F1 Score']:.4f} | Recall: {best_metrics['Recall']:.4f}")

        return best_model, best_metrics["F1 Score"], best_metrics.to_dict()

    def _normalize_metric_name(self):
        """
        Normalize the scoring metric to match known evaluation metrics.
        Used to validate the provided scoring string.
        """
        valid_metrics = ["Accuracy", "F1 Score", "ROC AUC", "Precision", "Recall"]
        for metric in valid_metrics:
            if self.scoring.lower() in metric.lower():
                return metric
        raise ValueError(f"Metric '{self.scoring}' is invalid. Use one of: {valid_metrics}")

# === External function to use from main.py ===
def run_grid_search_with_all_saves(model_class, data, param_grid, scoring, model_name, search_type="grid"):
    """
    Execute grid search and store all execution results.

    Parameters:
    - model_class: the model class to evaluate
    - data: dataset to train on
    - param_grid: dictionary of hyperparameters
    - scoring: metric to evaluate (default: F1)
    - model_name: name used for saving files
    - search_type: used for organizing result directories

    Returns:
    - best_model: trained model with best configuration
    - best_f1: best F1 score found
    - best_metrics: full metric dictionary of best configuration
    """
    f = io.StringIO()  # Capture all console output (optional)

    with redirect_stdout(f):
        gs = GridSearchWrapper(model_class, data, param_grid=param_grid, scoring=scoring)
        best_model, best_f1, best_metrics = gs.run()

    # Load the full results file
    all_results_file = f"results/metrics/{model_name}_grid_search_results.csv"
    if os.path.exists(all_results_file):
        all_results = pd.read_csv(all_results_file)
        os.makedirs(f"results/{search_type}_search/all_executions", exist_ok=True)

        # Save all results together
        all_results.to_csv(
            f"results/{search_type}_search/all_executions/{search_type}_search_{model_name}_all_results.csv",
            index=False
        )

        # Save each individual execution's results separately
        for idx, row in all_results.iterrows():
            execution_metrics = pd.DataFrame([row])
            execution_metrics.to_csv(
                f"results/{search_type}_search/all_executions/{search_type}_search_{model_name}_execution_{idx+1}_metrics.csv",
                index=False
            )

    return best_model, best_f1, best_metrics
