import os  # For file system operations
import pandas as pd  # For data manipulation
import joblib  # For saving models (not used here currently)
from sklearn.model_selection import train_test_split, TimeSeriesSplit  # For splitting data
from sklearn.base import BaseEstimator  # Base type for sklearn models

# Import metrics and visualization tools
from evaluation.metrics import compute_classification_metrics, print_classification_metrics
from evaluation.evaluation_utils import plot_all_evaluation_graphs


class BaseClassifierModel:
    """Base class wrapping common training and evaluation logic."""

    def __init__(
        self,
        model: BaseEstimator,  # A scikit-learn compatible model
        df: pd.DataFrame,  # DataFrame with features and target
        test_size=0.2,  # Fraction of data used for testing
        shuffle=False,  # Whether to shuffle the dataset
        use_sliding_window=False,  # Use time-series sliding window CV
        n_splits=5,  # Number of CV splits for sliding window
        **model_params,  # Additional parameters to save in metrics
    ):
        self.model = model
        self.df = df.copy()
        self.test_size = test_size
        self.shuffle = shuffle
        self.use_sliding_window = use_sliding_window
        self.n_splits = n_splits
        self.model_params = model_params
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.y_pred = None
        self.y_proba = None
        self.feature_names = None

    def prepare_data(self):
        """
        Split the dataset into train and test sets.
        Excludes 'Date', 'Ticker', and 'Target' columns from features.
        """
        X = self.df.drop(columns=["Date", "Ticker", "Target"])
        y = self.df["Target"]
        self.feature_names = X.columns.tolist()

        print("\nClass balance:")
        print(y.value_counts())  # Print class distribution

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, shuffle=self.shuffle
        )

    def train(self):
        """
        Train the model using a simple train/test split.
        Predicts both class labels and probabilities (if supported).
        """
        self.prepare_data()
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)

        if hasattr(self.model, "predict_proba"):
            self.y_proba = self.model.predict_proba(self.X_test)[:, 1]
        elif hasattr(self.model, "decision_function"):
            self.y_proba = self.model.decision_function(self.X_test)
        else:
            self.y_proba = None

    def train_sliding_window(self):
        """
        Train using time-series cross-validation (sliding window).
        Averages metrics over all splits.
        """
        self.df.sort_values("Date", inplace=True)  # Ensure chronological order
        X = self.df.drop(columns=["Date", "Ticker", "Target"])
        y = self.df["Target"]
        self.feature_names = X.columns.tolist()

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        metrics_sum = None

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            if hasattr(self.model, "predict_proba"):
                y_proba = self.model.predict_proba(X_test)[:, 1]
            elif hasattr(self.model, "decision_function"):
                y_proba = self.model.decision_function(X_test)
            else:
                y_proba = None

            metrics = compute_classification_metrics(y_test, y_pred, y_proba)

            if metrics_sum is None:
                metrics_sum = {k: 0 for k in metrics}
            for k, v in metrics.items():
                metrics_sum[k] += v

        avg_metrics = {k: v / self.n_splits for k, v in metrics_sum.items()}
        print_classification_metrics(avg_metrics)
        return avg_metrics

    def evaluate(self):
        """
        Evaluate the trained model and print metrics.
        Warns if the model predicts only one class.
        """
        if self.y_pred is not None and len(set(self.y_pred)) == 1:
            print("Warning: Model predicted a single class; metrics may be misleading.")

        metrics = compute_classification_metrics(self.y_test, self.y_pred, self.y_proba)
        print_classification_metrics(metrics)
        return metrics

    def save_metrics(self, metrics: dict):
        """
        Save metrics to CSV, including hyperparameters used.
        """
        os.makedirs("results/metrics", exist_ok=True)
        for param, value in self.model_params.items():
            metrics[param] = value

        metrics_df = pd.DataFrame([metrics])
        csv_path = os.path.join("results/metrics", f"{self.model.__class__.__name__}_metrics.csv")
        metrics_df.to_csv(csv_path, index=False)
        print(f"Metrics saved to {csv_path}")

    def run_all(self):
        """
        End-to-end pipeline: train → evaluate → visualize → save metrics.
        """
        print(f"\nTraining model {self.model.__class__.__name__}...")
        if self.use_sliding_window:
            metrics = self.train_sliding_window()
        else:
            self.train()
            metrics = self.evaluate()

            # Generate and save evaluation plots (only for train/test split)
            plot_all_evaluation_graphs(
                y_true=self.y_test,
                y_pred=self.y_pred,
                y_proba=self.y_proba,
                feature_importances=getattr(self.model, "feature_importances_", None),
                feature_names=self.feature_names,
                model_name=self.model.__class__.__name__,
                save_dir="results",
            )

        self.save_metrics(metrics)

    def get_metrics(self):
        """
        Return model evaluation metrics along with model parameters.
        """
        metrics = compute_classification_metrics(self.y_test, self.y_pred, self.y_proba)
        metrics["Model"] = self.model.__class__.__name__
        for param, value in self.model_params.items():
            metrics[param] = value
        return metrics

    def get_predictions(self):
        """
        Return only predicted class labels (from test set).
        """
        if self.y_pred is None:
            raise ValueError("Model has not been trained. Call run_all() first.")
        return self.y_pred

    def get_raw_predictions(self):
        """
        Return both true and predicted class labels (from test set).
        """
        if self.y_pred is None or self.y_test is None:
            raise ValueError("Model has not been trained. Call run_all() first.")
        return self.y_test, self.y_pred
