import os
import shutil
import pandas as pd

# === Módulos del proyecto ===
from ingest.stock_data_fetcher import StockDataFetcher
from preprocessing.data_preprocessor import DataPreprocessor
from feature_engineering.technical_indicators import TechnicalIndicators
from eda.stock_eda import StockEDA

from models.random_forest import RandomForestModel
from models.logistic_regression import LogisticModel
from models.xgboost_model import XGBoostClassifierModel
from models.svm_model import SVMClassifierModel
from models.grid_search_wrapper import GridSearchWrapper

# === Crear carpetas necesarias ===
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/preprocessed", exist_ok=True)
os.makedirs("results/eda", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/best_models", exist_ok=True)
os.makedirs("results/best_model", exist_ok=True)


# === 1. Obtener datos ===
csv_path = "data/raw/stock_data.csv"
if os.path.exists(csv_path):
    print(f"Archivo {csv_path} encontrado. Cargando datos desde CSV...")
    df = StockDataFetcher.load_from_csv(csv_path)
else:
    print(f"Archivo {csv_path} no encontrado. Descargando datos desde Yahoo Finance...")
    fetcher = StockDataFetcher(ticker=["AAPL", "BTC-USD", "GOOG"], start_date="2017-01-01")
    df = fetcher.fetch()
    fetcher.save_to_csv(csv_path)
    fetcher.quick_summary()

# === 2. Filtrar un solo activo (AAPL) ===
df = df[df["Ticker"] == "AAPL"].copy()

# === 3. Añadir indicadores técnicos ===
df = (
    TechnicalIndicators(df)
    .add_rsi()
    .add_macd()
    .add_bollinger_bands()
    .get_data()
    .dropna()
)

# === 4. Análisis exploratorio ===
eda = StockEDA(df)
eda.run_all(save_dir="results/eda")

# === 5. Preprocesamiento ===
forecast_horizon = 5
preprocessor = DataPreprocessor(
    data=df,
    scale="standard",
    model_type="classification",
    target_type="direction",
    forecast_horizon=forecast_horizon
)
df_processed = preprocessor.preprocess()

# === Guardar datos preprocesados ===
preprocessed_path = "data/preprocessed/stock_data_preprocessed.csv"
df_processed.to_csv(preprocessed_path, index=False)
print(f"Datos preprocesados guardados en {preprocessed_path}")

# === 6. Grid Search para cada modelo ===
results = []

print("\n=== Grid Search: Logistic Regression ===")
log_params = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["liblinear", "lbfgs"],
    "max_iter": [200, 500]
}
log_gs = GridSearchWrapper(LogisticModel, df_processed, param_grid=log_params, scoring="f1")
_, _, log_metrics = log_gs.run()
results.append(log_metrics)

print("\n=== Grid Search: Random Forest ===")
rf_params = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5, 10],
    "random_state": [42]
}
rf_gs = GridSearchWrapper(RandomForestModel, df_processed, param_grid=rf_params, scoring="f1")
_, _, rf_metrics = rf_gs.run()
results.append(rf_metrics)

print("\n=== Grid Search: XGBoost Classifier ===")
xgb_params = {
    "n_estimators": [100, 300],
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 6]
}
xgb_gs = GridSearchWrapper(XGBoostClassifierModel, df_processed, param_grid=xgb_params, scoring="f1")
_, _, xgb_metrics = xgb_gs.run()
results.append(xgb_metrics)

print("\n=== Grid Search: SVM Classifier ===")
svm_params = {
    "kernel": ["linear", "rbf"],
    "C": [0.1, 1, 10],
    "gamma": ["scale", "auto"]
}
svm_gs = GridSearchWrapper(SVMClassifierModel, df_processed, param_grid=svm_params, scoring="f1")
_, _, svm_metrics = svm_gs.run()
results.append(svm_metrics)

# === 7. Comparación de Resultados ===
results_df = pd.DataFrame(results)
print("\n=== Comparación de Modelos ===")
print(results_df.sort_values(by=["F1 Score", "Recall"], ascending=[False, False]).to_string(index=False))

# Guardar comparación en CSV
results_df.to_csv("results/metrics/model_comparison.csv", index=False)
print("Resultados guardados en results/metrics/model_comparison.csv")

# === 8. Guardar el mejor modelo global (prioridad: F1, luego Recall) ===
best_overall = results_df.sort_values(by=["F1 Score", "Recall"], ascending=[False, False]).iloc[0]
best_model_name = best_overall["Model"]
print(f"\nMejor modelo global: {best_model_name}")

# Copiar .pkl y .csv desde best_models a best_model
src_model = os.path.join("results/best_models", f"{best_model_name}.pkl")
dst_model = os.path.join("results/best_model", f"{best_model_name}.pkl")
src_metrics = os.path.join("results/best_models", f"{best_model_name}_metrics.csv")
dst_metrics = os.path.join("results/best_model", f"{best_model_name}_metrics.csv")

shutil.copy(src_model, dst_model)
shutil.copy(src_metrics, dst_metrics)

print("Modelo y métricas del mejor modelo global guardados en results/best_model/")
