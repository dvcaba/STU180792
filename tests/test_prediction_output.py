# tests/test_prediction_output.py

import pytest  # Pytest for fixtures and assertions
import pandas as pd  # For DataFrame handling
from preprocessing.data_preprocessor import DataPreprocessor  # Preprocessing pipeline
from models.random_forest import RandomForestModel  # Random Forest model
from ingest.stock_data_fetcher import StockDataFetcher  # For loading raw stock data
from feature_engineering.technical_indicators import TechnicalIndicators  # Technical indicators module

@pytest.fixture(scope="module")
def prediction_outputs():
    """
    Pytest fixture that sets up the complete pipeline to produce predictions:
    - Loads data from CSV
    - Applies technical indicators
    - Preprocesses data
    - Trains Random Forest model
    - Generates and returns predictions
    Returns:
        Tuple: (processed DataFrame, predictions array)
    """
    # Load raw historical stock data from file
    df = StockDataFetcher.load_from_csv("data/raw/stock_data.csv")
    
    # Apply technical indicators
    df = (
        TechnicalIndicators(df)
        .add_rsi()                 # Add RSI column
        .add_macd()                # Add MACD-related columns
        .add_bollinger_bands()     # Add Bollinger Bands
        .get_data()
        .dropna()                  # Remove rows with NaN values from indicator calculations
    )

    # Preprocess the data (feature scaling and target generation)
    preprocessor = DataPreprocessor(
        data=df,
        scale="standard",            # Use StandardScaler
        model_type="classification", # Classification task
        target_type="direction",     # Predict up/down movement
        forecast_horizon=3           # Predict 3 periods ahead
    )
    df_processed = preprocessor.preprocess()

    # Train a Random Forest model
    model = RandomForestModel(df_processed)
    model.run_all()

    # Generate predictions on the test set
    predictions = model.get_predictions()
    return df_processed, predictions

def test_predictions_not_empty(prediction_outputs):
    """
    Ensure that the model generates predictions and the output is not empty.
    """
    _, predictions = prediction_outputs
    assert predictions is not None and len(predictions) > 0, "No predictions generated."

def test_prediction_length(prediction_outputs):
    """
    Validate that the number of predictions is approximately 20% of the full dataset size.
    """
    df_processed, predictions = prediction_outputs
    test_size = int(0.2 * len(df_processed))  # Expected number of test samples (20% split)
    assert abs(len(predictions) - test_size) <= 1, "Prediction length mismatch."

def test_prediction_value_range(prediction_outputs):
    """
    Ensure that prediction values are valid for a binary classification task (0 or 1).
    """
    _, predictions = prediction_outputs
    unique_preds = set(predictions)
    assert unique_preds.issubset({0, 1}), f"Unexpected prediction values: {unique_preds}"
