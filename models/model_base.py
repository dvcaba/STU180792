import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator

from evaluation.metrics import compute_classification_metrics, print_classification_metrics
from evaluation.evaluation_utils import plot_all_evaluation_graphs


class BaseClassifierModel:
    def __init__(self, model: BaseEstimator, df: pd.DataFrame, test_size=0.2, shuffle=False, **model_params):
        self.model = model
        self.df = df.copy()
        self.test_size = test_size
        self.shuffle = shuffle
        self.model_params = model_params
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.y_pred = None
        self.y_proba = None
        self.feature_names = None

    def prepare_data(self):
        X = self.df.drop(columns=["Date", "Ticker", "Target"])
        y = self.df["Target"]
        self.feature_names = X.columns.tolist()

        print("\nBalance de clases:")
        print(y.value_counts())

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, shuffle=self.shuffle
        )

    def train(self):
        self.prepare_data()
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)

        if hasattr(self.model, "predict_proba"):
            self.y_proba = self.model.predict_proba(self.X_test)[:, 1]
        elif hasattr(self.model, "decision_function"):
            self.y_proba = self.model.decision_function(self.X_test)
        else:
            self.y_proba = None

    def evaluate(self):
        metrics = compute_classification_metrics(self.y_test, self.y_pred, self.y_proba)
        print_classification_metrics(metrics)
        return metrics

    def save_metrics(self, metrics: dict):
        os.makedirs("results/metrics", exist_ok=True)
        for param, value in self.model_params.items():
            metrics[param] = value

        metrics_df = pd.DataFrame([metrics])
        csv_path = os.path.join("results/metrics", f"{self.model.__class__.__name__}_metrics.csv")
        metrics_df.to_csv(csv_path, index=False)
        print(f"Métricas guardadas en {csv_path}")

    def run_all(self):
        print(f"\nEntrenando modelo {self.model.__class__.__name__}...")
        self.train()
        metrics = self.evaluate()

        # Guardar gráficos de evaluación
        plot_all_evaluation_graphs(
            y_true=self.y_test,
            y_pred=self.y_pred,
            y_proba=self.y_proba,
            feature_importances=getattr(self.model, "feature_importances_", None),
            feature_names=self.feature_names,
            model_name=self.model.__class__.__name__,
            save_dir="results"
        )

        # Ya no se guarda en results/models/
        # self.save_model()
        self.save_metrics(metrics)

    def get_metrics(self):
        metrics = compute_classification_metrics(self.y_test, self.y_pred, self.y_proba)
        metrics["Model"] = self.model.__class__.__name__
        for param, value in self.model_params.items():
            metrics[param] = value
        return metrics
