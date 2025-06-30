# tests/test_feature_engineering.py

import pytest  # Pytest framework for testing
import pandas as pd  # For DataFrame manipulation
from ingest.stock_data_fetcher import StockDataFetcher  # Module to fetch stock data
from feature_engineering.technical_indicators import TechnicalIndicators  # Module to compute indicators

@pytest.fixture(scope="module")
def engineered_data():
    """
    Fixture that loads raw stock data and applies technical indicators:
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands
    The result is cleaned with dropna to ensure indicators are fully computed.
    """
    df = StockDataFetcher.load_from_csv("data/raw/stock_data.csv")  # Load raw stock data from CSV
    df = (
        TechnicalIndicators(df)  # Initialize indicator pipeline
        .add_rsi()               # Add RSI column
        .add_macd()              # Add MACD, Signal, Histogram
        .add_bollinger_bands()   # Add Bollinger Bands
        .get_data()              # Retrieve the updated DataFrame
    )
    return df.dropna()  # Drop rows with NaN (incomplete indicator computation)

def test_technical_indicators_exist(engineered_data):
    """
    Test that the DataFrame contains all expected technical indicators after feature engineering.
    """
    required_indicators = [
        "RSI", "MACD", "MACD_Signal", "MACD_Hist", 
        "BB_Middle", "BB_Upper", "BB_Lower"
    ]
    for col in required_indicators:
        assert col in engineered_data.columns, f"Missing indicator: {col}"

def test_no_nan_in_indicators(engineered_data):
    """
    Test that there are no NaN values in any of the technical indicator columns
    after calling dropna in the fixture.
    """
    indicator_cols = [
        "RSI", "MACD", "MACD_Signal", "MACD_Hist", 
        "BB_Middle", "BB_Upper", "BB_Lower"
    ]
    for col in indicator_cols:
        assert not engineered_data[col].isnull().any(), f"NaNs remain in {col}"

def test_alignment_of_targets(engineered_data):
    """
    Ensure proper ordering of rows after feature engineering:
    - The "Date" column must exist.
    - Each ticker's data must be sorted by date.
    - The entire DataFrame index must be sorted.
    """
    assert "Date" in engineered_data.columns, "Date column missing"

    for ticker, dates in engineered_data.groupby("Ticker")["Date"]:
        assert dates.is_monotonic_increasing, f"Dates not sorted for {ticker}"

    assert engineered_data.index.is_monotonic_increasing, "Index not sorted"
