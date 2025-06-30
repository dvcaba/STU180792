# === Standard library imports ===
import os  # Provides functions to interact with the operating system
import shutil  # Used to copy and remove files or directories
import pandas as pd  # For data manipulation and analysis
import io  # For managing input/output streams
import sys  # System-specific parameters and functions
from contextlib import redirect_stdout  # Temporarily redirects standard output

# === Project-specific module imports ===
from ingest.stock_data_fetcher import StockDataFetcher  # Handles fetching stock data
from preprocessing.data_preprocessor import DataPreprocessor  # Preprocesses the dataset
from feature_engineering.technical_indicators import TechnicalIndicators  # Adds technical indicators to the data
from eda.stock_eda import StockEDA  # Performs exploratory data analysis

# Import machine learning model wrappers
from models.random_forest import RandomForestModel
from models.logistic_regression import LogisticModel
from models.xgboost_model import XGBoostClassifierModel
from models.svm_model import SVMClassifierModel

# Import evaluation tools
from evaluation.metrics import save_and_display_results  # Saves and displays performance metrics
from evaluation.evaluation_utils import rename_model_plots  # Renames plot files for easier comparison

# Import scikit-learn and joblib for model training and saving
from sklearn.model_selection import train_test_split  # Splits dataset into train/test sets
import joblib  # Saves and loads trained models to/from disk

# Hyperparameter tuning utilities
from models.grid_search_wrapper import run_grid_search_with_all_saves  # Grid search with logging and saving
from models.random_search_wrapper import run_random_search_with_all_saves  # Randomized search with logging and saving

# === Miscellaneous setup ===
import warnings
warnings.filterwarnings('ignore')  # Ignore all warnings to keep output clean

import random
import numpy as np
import yaml  # For loading configuration files

# === Set global seed for reproducibility ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# === Load hyperparameter configuration file ===
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# === Extract pipeline-level configuration ===
pipeline_cfg = config.get("pipeline", {})
TICKERS = pipeline_cfg.get("tickers", ["AAPL", "BTC-USD", "^GSPC"])  # Default tickers
START_DATE = pipeline_cfg.get("start_date", "2017-01-01")  # Default start date
FILTER_TICKER = pipeline_cfg.get("filter_ticker", "^GSPC")  # Main ticker to focus on
FORECAST_HORIZON = pipeline_cfg.get("forecast_horizon", 5)  # Number of days to forecast

# === Create necessary folders if they don't exist ===
for path in [
    "data/raw", "data/preprocessed",
    "results/eda",
    "results/grid_search/best_models", "results/grid_search/metrics", 
    "results/grid_search/plots", "results/grid_search/all_executions",
    "results/random_search/best_models", "results/random_search/metrics",
    "results/random_search/plots", "results/random_search/all_executions",
    "results/global_best", "results/global_comparison"
    ]:
    os.makedirs(path, exist_ok=True)

# === Step 1: Fetch or load stock data ===
print("\n=== Data Loading ===")
csv_path = "data/raw/stock_data.csv"
if os.path.exists(csv_path):
    # If data already exists locally, load it
    print(f"Found {csv_path}. Loading data from CSV...")
    df = StockDataFetcher.load_from_csv(csv_path)
else:
    # If not, fetch it from Yahoo Finance and save it
    print(f"File {csv_path} not found. Downloading data from Yahoo Finance...")
    fetcher = StockDataFetcher(ticker=TICKERS, start_date=START_DATE)
    df = fetcher.fetch()
    fetcher.save_to_csv(csv_path)
    fetcher.quick_summary()

# Display head and columns of the ingested data
print("\n=== Ingest: Head and Columns ===")
print(df.head())
print("Columnas:", df.columns.tolist())

# === Step 2: Filter data for a specific ticker ===
df = df[df["Ticker"] == FILTER_TICKER].copy()
print(f"Filtered data to {FILTER_TICKER} only")

# === Step 3: Add technical indicators ===
print("\n=== Adding Technical Indicators ===")
df = (
    TechnicalIndicators(df)
    .add_rsi()  # Add Relative Strength Index
    .add_macd()  # Add Moving Average Convergence Divergence
    .add_bollinger_bands()  # Add Bollinger Bands
    .get_data()
    .dropna()  # Drop rows with missing values
)
print("Added RSI, MACD, and Bollinger Bands")
print("\n=== Feature Engineering: Head and Columns ===")
print(df.head())
print("Columnas:", df.columns.tolist())

# === Step 4: Perform exploratory data analysis (EDA) ===
print("\n=== Running Exploratory Data Analysis ===")
eda = StockEDA(df)
eda.run_all(save_dir="results/eda")  # Save all EDA plots
print("EDA plots saved to results/eda/")
print("Columns analyzed EDA:", df.columns.tolist())

# === Step 5: Preprocess the data ===
print("\n=== Preprocessing Data ===")
preprocessor = DataPreprocessor(
    data=df,
    scale="standard",  # StandardScaler for feature scaling
    model_type="classification",  # Classification task
    target_type="direction",  # Predict direction of price movement
    forecast_horizon=FORECAST_HORIZON  # Number of days to predict ahead
)
df_processed = preprocessor.preprocess()
df_processed.to_csv("data/preprocessed/stock_data_preprocessed.csv", index=False)
print("\n=== Processed: Head and Columns ===")
print(df_processed.head())
print("Columns:", df_processed.columns.tolist())

# === Step 6: Train baseline model (Logistic Regression) ===
print("\n=== Training Baseline Model ===")
f = io.StringIO()
with redirect_stdout(f):  # Suppress training output
    baseline_model = LogisticModel(df_processed)
    baseline_model.run_all()

# Load and save baseline metrics
baseline_metrics = pd.read_csv("results/metrics/LogisticRegression_metrics.csv")
baseline_metrics.to_csv("results/global_comparison/baseline_LogisticRegression_metrics.csv", index=False)

# Display baseline results
save_and_display_results(
    baseline_metrics,
    "results/global_comparison/baseline_results.csv",
    "BASELINE MODEL RESULTS"
)

# === Step 7: Grid search for hyperparameter tuning ===
print("\n" + "="*80)
print("STARTING GRID SEARCH PHASE")
print("="*80)

grid_search_results = []

# Logistic Regression
log_params = config["grid_search"]["logistic_regression"]
_, _, log_metrics = run_grid_search_with_all_saves(LogisticModel, df_processed, log_params, "f1", "LogisticRegression", "grid")
grid_search_results.append(log_metrics)

# Random Forest
rf_params = config["grid_search"]["random_forest"]
_, _, rf_metrics = run_grid_search_with_all_saves(RandomForestModel, df_processed, rf_params, "f1", "RandomForest", "grid")
grid_search_results.append(rf_metrics)

# XGBoost
xgb_params = config["grid_search"]["xgboost"]
_, _, xgb_metrics = run_grid_search_with_all_saves(XGBoostClassifierModel, df_processed, xgb_params, "f1", "XGBoost", "grid")
grid_search_results.append(xgb_metrics)

# Support Vector Machine
svm_params = config["grid_search"]["svm"]
_, _, svm_metrics = run_grid_search_with_all_saves(SVMClassifierModel, df_processed, svm_params, "f1", "SVM", "grid")
grid_search_results.append(svm_metrics)

# Save and display grid search comparison
grid_results_df = pd.DataFrame(grid_search_results)
save_and_display_results(
    grid_results_df,
    "results/grid_search/metrics/grid_search_best_models_comparison.csv",
    "GRID SEARCH - BEST MODEL OF EACH TYPE"
)

# Print best hyperparameters per model from grid search
print("\n--- Best hiperparameters found with GRID SEARCH ---")
for _, row in grid_results_df.iterrows():
    model_name = row["Model"]
    param_keys = [k for k in row.index if k not in ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]]
    param_values = {k: row[k] for k in param_keys if pd.notnull(row[k])}
    print(f"Grid Search ({model_name}): {param_values}")

# === Step 8: Random search for hyperparameter tuning ===
print("\n" + "="*80)
print("STARTING RANDOM SEARCH PHASE")
print("="*80)

random_search_results = []

# Logistic Regression
log_distributions = config["random_search"]["logistic_regression"]
_, _, log_rs_metrics = run_random_search_with_all_saves(LogisticModel, df_processed, log_distributions, 10, "f1", "LogisticRegression")
random_search_results.append(log_rs_metrics)

# Random Forest
rf_distributions = config["random_search"]["random_forest"]
_, _, rf_rs_metrics = run_random_search_with_all_saves(RandomForestModel, df_processed, rf_distributions, 15, "f1", "RandomForest")
random_search_results.append(rf_rs_metrics)

# XGBoost
xgb_distributions = config["random_search"]["xgboost"]
_, _, xgb_rs_metrics = run_random_search_with_all_saves(XGBoostClassifierModel, df_processed, xgb_distributions, 20, "f1", "XGBoost")
random_search_results.append(xgb_rs_metrics)

# Support Vector Machine
svm_distributions = config["random_search"]["svm"]
_, _, svm_rs_metrics = run_random_search_with_all_saves(SVMClassifierModel, df_processed, svm_distributions, 15, "f1", "SVM")
random_search_results.append(svm_rs_metrics)

# Save and display random search comparison
random_results_df = pd.DataFrame(random_search_results)
save_and_display_results(
    random_results_df,
    "results/random_search/metrics/random_search_best_models_comparison.csv",
    "RANDOM SEARCH - BEST MODEL OF EACH TYPE"
)

# Print best hyperparameters per model from random search
print("\n--- Best hiperparameters found with RANDOM SEARCH ---")
for _, row in random_results_df.iterrows():
    model_name = row["Model"]
    param_keys = [k for k in row.index if k not in ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]]
    param_values = {k: row[k] for k in param_keys if pd.notnull(row[k])}
    print(f"Random Search ({model_name}): {param_values}")

# === Step 9: Compare best grid vs random per model ===
print("\n" + "="*80)
print("GLOBAL COMPARISON - BEST OF EACH MODEL TYPE")
print("="*80)

model_types = ["LogisticRegression", "RandomForestClassifier", "XGBClassifier", "SVC"]
global_best_models = []

# For each model type, compare the best from grid vs random search
for model_type in model_types:
    grid_model = grid_results_df[grid_results_df["Model"] == model_type]
    random_model = random_results_df[random_results_df["Model"] == model_type]
    
    if not grid_model.empty and not random_model.empty:
        if grid_model.iloc[0]["F1 Score"] >= random_model.iloc[0]["F1 Score"]:
            best = grid_model.iloc[0].copy()
            best["Search Method"] = "Grid Search"
        else:
            best = random_model.iloc[0].copy()
            best["Search Method"] = "Random Search"
        global_best_models.append(best.to_dict())

# Save best version of each model type
global_comparison_df = pd.DataFrame(global_best_models)
save_and_display_results(
    global_comparison_df[["Model", "Search Method", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]],
    "results/global_comparison/global_best_of_each_model_type.csv",
    "GLOBAL - BEST VERSION OF EACH MODEL TYPE"
)

# === Step 10: Identify overall best model ===
all_results = grid_search_results + random_search_results
all_results_df = pd.DataFrame(all_results)

# Sort by F1 Score and Recall
best_overall = all_results_df.sort_values(by=["F1 Score", "Recall"], ascending=[False, False]).iloc[0]
best_model_name = best_overall["Model"]

# Display best model info
print(f"\n{'='*80}")
print(f"*** GLOBAL BEST MODEL ***")
print(f"Model: {best_model_name}")
print(f"F1 Score: {best_overall['F1 Score']:.4f}")
print(f"Recall: {best_overall['Recall']:.4f}")
print(f"Accuracy: {best_overall['Accuracy']:.4f}")
print(f"Precision: {best_overall['Precision']:.4f}")
print(f"ROC AUC: {best_overall['ROC AUC']:.4f}")

# Extract hyperparameters from best model
standard_metrics = {"Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "Search Method"}
hyperparams = {
    k: (float(v) if isinstance(v, (np.floating, float)) else v)
    for k, v in best_overall.items()
    if k not in standard_metrics and pd.notnull(v)
}
print("Hyperparameters:", hyperparams)
print(f"{'='*80}")

# Save best model info to disk
pd.DataFrame([best_overall]).to_csv(
    "results/global_best/global_best_model_info.csv", 
    index=False
)

# Copy model and metrics files to global_best folder
src_model = os.path.join("results/best_models", f"{best_model_name}.pkl")
dst_model = os.path.join("results/global_best", f"global_best_{best_model_name}.pkl")
src_metrics = os.path.join("results/best_models", f"{best_model_name}_metrics.csv")
dst_metrics = os.path.join("results/global_best", f"global_best_{best_model_name}_metrics.csv")

if os.path.exists(src_model):
    shutil.copy2(src_model, dst_model)
    shutil.copy2(src_metrics, dst_metrics)

# Copy plots (confusion matrix, ROC, feature importance)
source_dirs = ["results/confusion_matrices", "results/roc_curves", "results/feature_importance"]
rename_model_plots(best_model_name, source_dirs, "results/global_best", "global_best")

# === Step 11: Generate predictions using the best model ===
best_model_path = os.path.join("results/best_models", f"{best_model_name}.pkl")
if os.path.exists(best_model_path):
    loaded_model = joblib.load(best_model_path)
    features = df_processed.drop(columns=["Date", "Ticker", "Target"])
    target = df_processed["Target"]
    _, X_test, _, _ = train_test_split(features, target, test_size=0.2, shuffle=False)
    preds = loaded_model.predict(X_test)
    os.makedirs("results", exist_ok=True)
    pred_path = os.path.join("results", "best_model_predictions.csv")
    pd.DataFrame(preds, columns=["Prediction"]).to_csv(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")
    print(preds)

else:
    print(f"Best model file {best_model_path} not found. Skipping prediction step.")

# === Final summary with key result files ===
print("\n" + "="*80)
print("\nKey files:")
print(f"- Baseline: results/global_comparison/baseline_results.csv")
print(f"- Best Grid Search: results/grid_search/metrics/grid_search_overall_best.csv")
print(f"- Best Random Search: results/random_search/metrics/random_search_overall_best.csv")
print(f"- Global Best: results/global_best/global_best_model_info.csv")
