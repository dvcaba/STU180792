# tests/test_pipeline_integration.py

import pytest  # Pytest for testing
from ingest.stock_data_fetcher import StockDataFetcher  # Module for loading raw data
from feature_engineering.technical_indicators import TechnicalIndicators  # Feature engineering pipeline
from preprocessing.data_preprocessor import DataPreprocessor  # Data preprocessing module
from models.xgboost_model import XGBoostClassifierModel  # Model to be tested

@pytest.mark.integration
def test_full_pipeline_execution():
    """
    Integration test for the full machine learning pipeline:
    - Data ingestion
    - Feature engineering
    - Preprocessing
    - Model training and prediction
    This test ensures all pipeline stages work together without errors.
    """
    try:
        # Step 1: Ingest data from local CSV (offline source)
        df = StockDataFetcher.load_from_csv("data/raw/stock_data.csv")

        # Step 2: Add technical indicators (RSI, MACD, Bollinger Bands)
        df = (
            TechnicalIndicators(df)      # Initialize the indicator engine
            .add_rsi()                   # Add Relative Strength Index
            .add_macd()                  # Add MACD, MACD Signal, MACD Histogram
            .add_bollinger_bands()       # Add Bollinger Bands (upper/lower/middle)
            .get_data()                  # Retrieve the updated DataFrame
            .dropna()                    # Drop rows with NaNs (from indicator calculations)
        )

        # Step 3: Preprocess the data
        preprocessor = DataPreprocessor(
            data=df,
            scale="standard",            # StandardScaler for feature scaling
            model_type="classification", # Prepare data for classification task
            target_type="direction",     # Predict up/down movement (binary classification)
            forecast_horizon=3           # Look ahead 3 periods
        )
        df_processed = preprocessor.preprocess()  # Output cleaned and scaled data

        # Step 4: Train the model and generate predictions
        model = XGBoostClassifierModel(df_processed)  # Initialize model with processed data
        model.run_all()                               # Run train/test pipeline
        preds = model.get_predictions()               # Get predictions from test set

        # Final assertion: ensure predictions are returned and not empty
        assert preds is not None and len(preds) > 0, "No predictions generated."
        print("Full pipeline executed successfully.")  # Log success

    except Exception as e:
        # If any stage fails, mark test as failed with the exception message
        pytest.fail(f"Pipeline integration failed: {e}")
