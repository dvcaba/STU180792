# tests/test_data_ingestion.py

import pytest  # Pytest framework for testing
import pandas as pd  # For DataFrame operations
from ingest.stock_data_fetcher import StockDataFetcher  # Import the stock data loader

@pytest.fixture(scope="module")
def fetched_data():
    """
    Pytest fixture to load sample stock data once per module.
    The data is loaded from a static CSV file for consistency in tests.
    """
    df = StockDataFetcher.load_from_csv("data/raw/stock_data.csv")  # Load pre-saved CSV file
    return df

def test_data_is_dataframe(fetched_data):
    """
    Test to verify that the returned object is a non-empty DataFrame.
    """
    assert isinstance(fetched_data, pd.DataFrame), "Data is not a DataFrame"
    assert not fetched_data.empty, "Fetched data is empty"

def test_required_columns_exist(fetched_data):
    """
    Check that all required columns exist in the fetched dataset.
    These are standard OHLCV (Open, High, Low, Close, Volume) and Ticker.
    """
    required_columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]
    for col in required_columns:
        assert col in fetched_data.columns, f"Missing required column: {col}"

def test_no_full_column_nulls(fetched_data):
    """
    Ensure that no column in the dataset is entirely null.
    """
    for col in fetched_data.columns:
        assert fetched_data[col].isnull().sum() < len(fetched_data), f"Column {col} is completely null"

def test_duplicate_rows(fetched_data):
    """
    Verify that the dataset does not contain fully duplicated rows.
    """
    duplicates = fetched_data.duplicated().sum()
    assert duplicates == 0, f"Found {duplicates} duplicated rows"

def test_no_duplicate_date_ticker_pairs(fetched_data):
    """
    Ensure there are no duplicate rows for the same (Date, Ticker) combination.
    This is crucial for time series integrity.
    """
    duplicates = fetched_data[["Date", "Ticker"]].duplicated().any()
    assert not duplicates, "Duplicate rows found for (Date, Ticker)"

def test_data_sorted_by_ticker_and_date(fetched_data):
    """
    Check that the DataFrame is sorted first by Ticker, then by Date.
    """
    sorted_df = fetched_data.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(sorted_df, fetched_data.reset_index(drop=True))

def test_dates_sorted_within_each_ticker(fetched_data):
    """
    Within each Ticker, dates should appear in strictly increasing order.
    """
    for ticker, dates in fetched_data.groupby("Ticker")["Date"]:
        assert dates.is_monotonic_increasing, f"Dates are not sorted for ticker: {ticker}"
