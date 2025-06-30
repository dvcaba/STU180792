import os  # For file and directory operations
import pandas as pd  # For handling dataframes
import joblib  # For saving trained models
from sklearn.model_selection import ParameterSampler  # For random hyperparameter search
import numpy as np  # For numerical operations (used in parameter cleaning)
import io  # For capturing output
from contextlib import redirect_stdout  # To redirect stdout to a variable (used in silent execution)

class RandomSearchWrapper:
    def __init__(self, model_class, data, param_distributions, n_iter=10, scoring="f1"):
        """
        Initialize the random search wrapper.

        Parameters:
        - model_class: the class of the model to be trained and evaluated
        - data: the dataset to use for training and evaluation
        - param_distributions: dictionary of parameter distributions for sampling
        - n_iter: number of parameter combinations to try (default: 10)
        - scoring: metric to use for evaluating models (default: F1)
        """
        self.model_class = model_class
        self.data = data
        self.param_list = list(ParameterSampler(param_distributions, n_iter=n_iter))  # Generate parameter samples
        self.scoring = scoring
        self.results = []  # Store results for all executions

    def run(self):
        """
        Run the random search: train and evaluate models with each sampled parameter set,
        save metrics, identify the best configuration, and save the best model and its metrics.
        """
        print(f"\n==> Starting Random Search for model: {self.model_class.__name__}")

        for i, params in enumerate(self.param_list):
            print(f"\n→ Random Search {i+1}/{len(self.param_list)}: {params}")
            cleaned_params = self._clean_parameters(params)  # Clean parameter values for compatibility

            try:
                model = self.model_class(self.data, **cleaned_params)  # Instantiate model
                model.run_all()  # Train and evaluate model
                metrics = model.get_metrics()  # Get evaluation metrics

                # Combine metrics and parameter values into a single result dictionary
                metrics_with_params = metrics.copy()
                metrics_with_params.update(cleaned_params)
                self.results.append(metrics_with_params)
            except Exception as e:
                print(f"Error with parameters {cleaned_params}: {e}")
                continue

        if not self.results:
            raise ValueError("No successful model runs in random search")

        # Save all results to CSV
        os.makedirs("results/metrics", exist_ok=True)
        results_df = pd.DataFrame(self.results)
        model_name = self.model_class.__name__.replace("Model", "")
        csv_path = f"results/metrics/{model_name}_random_search_results.csv"

        try:
            results_df.to_csv(csv_path, index=False)
            print(f"All random search results saved to {csv_path}")
        except Exception as e:
            print(f"Error saving results to {csv_path}: {e}")

        # Identify the best configuration based on F1 score and then Recall
        results_df_sorted = results_df.sort_values(by=["F1 Score", "Recall"], ascending=[False, False])
        best_metrics = results_df_sorted.iloc[0]
        best_params = {k: best_metrics[k] for k in self.param_list[0].keys() if k in best_metrics}
        best_params = self._clean_parameters(best_params)

        # Retrain the best model
        best_model = self.model_class(self.data, **best_params)
        best_model.run_all()

        # Save the best model and its metrics
        os.makedirs("results/best_models", exist_ok=True)
        model_class_name = best_model.model.__class__.__name__
        best_model_path = os.path.join("results/best_models", f"{model_class_name}.pkl")
        best_metrics_path = os.path.join("results/best_models", f"{model_class_name}_metrics.csv")

        try:
            joblib.dump(best_model.model, best_model_path)
            print(f"Best model of type {model_class_name} saved to {best_model_path}")
        except Exception as e:
            print(f"Error saving model {model_class_name}: {e}")

        try:
            pd.DataFrame([best_metrics]).to_csv(best_metrics_path, index=False)
            print(f"Best model metrics saved to {best_metrics_path}")
        except Exception as e:
            print(f"Error saving metrics for model {model_class_name}: {e}")

        print("\n=== Best configuration found ===")
        print(f"Parameters: {best_params}")
        print(f"F1 Score: {best_metrics['F1 Score']:.4f} | Recall: {best_metrics['Recall']:.4f}")

        return best_model, best_metrics["F1 Score"], best_metrics.to_dict()

    def _clean_parameters(self, params):
        """
        Clean parameter dictionary:
        - Convert NaNs to None
        - Convert float integers to int for known integer hyperparameters
        """
        cleaned = {}
        for k, v in params.items():
            if isinstance(v, float) and np.isnan(v):
                cleaned[k] = None
            elif isinstance(v, float) and v.is_integer():
                int_params = ['n_estimators', 'max_depth', 'min_samples_split',
                              'min_samples_leaf', 'max_iter', 'degree', 'n_jobs']
                if any(param in k for param in int_params):
                    cleaned[k] = int(v)
                else:
                    cleaned[k] = v
            else:
                cleaned[k] = v
        return cleaned

# === External function to be imported from main.py ===
def run_random_search_with_all_saves(model_class, data, param_distributions, n_iter, scoring, model_name):
    """
    Execute a full random search pipeline, saving all execution details and best model.

    Parameters:
    - model_class: class of model to train
    - data: training dataset
    - param_distributions: dictionary of parameter distributions
    - n_iter: number of parameter combinations to sample
    - scoring: metric to select the best model
    - model_name: name used for output files
    """
    f = io.StringIO()  # Redirect output (silently or to file if needed)

    with redirect_stdout(f):
        rs = RandomSearchWrapper(model_class, data, param_distributions=param_distributions,
                                 n_iter=n_iter, scoring=scoring)
        best_model, best_f1, best_metrics = rs.run()

    # Load and save all results from CSV
    all_results_file = f"results/metrics/{model_name}_random_search_results.csv"
    if os.path.exists(all_results_file):
        all_results = pd.read_csv(all_results_file)
        os.makedirs("results/random_search/all_executions", exist_ok=True)

        # Save all results
        all_results.to_csv(
            f"results/random_search/all_executions/random_search_{model_name}_all_results.csv",
            index=False
        )

        # Save individual executions
        for idx, row in all_results.iterrows():
            execution_metrics = pd.DataFrame([row])
            execution_metrics.to_csv(
                f"results/random_search/all_executions/random_search_{model_name}_execution_{idx+1}_metrics.csv",
                index=False
            )

    return best_model, best_f1, best_metrics
