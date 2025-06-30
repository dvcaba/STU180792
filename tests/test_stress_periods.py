# tests/test_stress_periods.py

import pytest  # Pytest for test decoration and assertions
import pandas as pd  # For handling dataframes
from ingest.stock_data_fetcher import StockDataFetcher  # Module to load raw stock data
from feature_engineering.technical_indicators import TechnicalIndicators  # Add indicators
from preprocessing.data_preprocessor import DataPreprocessor  # Preprocessing pipeline
from models.random_forest import RandomForestModel  # Model to test under stress

@pytest.mark.performance
def test_model_during_high_volatility_period():
    """
    Test the model's robustness during known periods of high market volatility.
    Example: October 2022 (rate hike decisions) or similar macroeconomic stress windows.
    """

    # Define the stress test period (e.g., market volatility due to macroeconomic events)
    stress_period = ("2022-10-01", "2022-11-15")

    # Warm-up period start date for sufficient historical context in indicators
    warmup_start = "2022-07-01"

    # Step 1: Load historical stock data from a local CSV file
    df = StockDataFetcher.load_from_csv("data/raw/stock_data.csv")

    # Filter data to include only the warm-up and stress periods
    df = df[(df["Date"] >= warmup_start) & (df["Date"] <= stress_period[1])]

    # Step 2: Apply technical indicators (RSI, MACD, Bollinger Bands)
    df = (
        TechnicalIndicators(df)
        .add_rsi()
        .add_macd()
        .add_bollinger_bands()
        .get_data()
        .dropna()  # Remove rows with incomplete indicator values
    )

    # Step 3: Preprocess the data for classification with a forecast horizon of 1
    preprocessor = DataPreprocessor(
        data=df,
        scale="standard",             # Standard scaling of features
        model_type="classification",  # Classification task
        target_type="direction",      # Predict price direction
        forecast_horizon=1,           # Predict 1 day ahead
        remove_outliers=False         # Do not remove outliers in this test
    )
    df_processed = preprocessor.preprocess()

    # Further filter the data to isolate only the stress test window
    df_processed = df_processed[
        (df_processed["Date"] >= stress_period[0]) &
        (df_processed["Date"] <= stress_period[1])
    ]

    # Step 4: Train a Random Forest model and run full training + evaluation
    model = RandomForestModel(df_processed)
    model.run_all()
    metrics = model.get_metrics()

    # Step 5: Assert the model does not collapse under stress (minimal threshold > 0.4)
    assert metrics["F1 Score"] > 0.4, f"Low F1 Score during stress: {metrics['F1 Score']}"
    assert metrics["Recall"] > 0.4, f"Low Recall during stress: {metrics['Recall']}"

    # Optional: log test success metrics
    print("Stress test passed with metrics:", metrics)
