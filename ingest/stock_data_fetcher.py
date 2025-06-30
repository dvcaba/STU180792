import yfinance as yf  # For downloading historical stock data
import pandas as pd  # For data manipulation
from datetime import datetime  # To handle dates
from typing import Union, List  # For type annotations

class StockDataFetcher:
    def __init__(self, ticker: Union[str, List[str]], start_date: str = "2020-01-01", end_date: str = None):
        """
        Initialize the fetcher with one or more stock tickers and a date range.
        
        Parameters:
        - ticker: single ticker as string or multiple tickers as a list of strings
        - start_date: start date for historical data (default: "2020-01-01")
        - end_date: end date for historical data (default: today)
        """
        if isinstance(ticker, str):
            self.tickers = [ticker.upper()]  # Convert to uppercase and wrap in list
        elif isinstance(ticker, list):
            self.tickers = [t.upper() for t in ticker]  # Convert all tickers to uppercase
        else:
            raise ValueError("Ticker must be a string or list of strings.")

        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.today().strftime('%Y-%m-%d')  # Default to today
        self.data = None  # Will hold the downloaded DataFrame

    def fetch(self) -> pd.DataFrame:
        """
        Download historical stock data using yfinance for all tickers.
        Returns a combined DataFrame with all tickers' data.
        """
        all_data = []  # Store data from each ticker

        for ticker in self.tickers:
            print(f"Fetching: {ticker}")
            try:
                # Download historical data for the current ticker
                df = yf.download(
                    ticker,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    auto_adjust=True  # Adjust prices for dividends/splits
                )

                if df.empty:
                    print(f"Warning: No data for {ticker}")
                    continue

                # Flatten multi-index columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]

                df = df.reset_index()  # Move Date from index to column
                df["Ticker"] = ticker  # Add column to identify the ticker
                all_data.append(df)

            except Exception as e:
                print(f"Error fetching {ticker}: {e}")

        if not all_data:
            raise ValueError("No data retrieved for any tickers.")

        self.data = pd.concat(all_data, ignore_index=True)  # Combine all data into one DataFrame
        return self.data

    def get_data(self) -> pd.DataFrame:
        """
        Return the fetched data. Raises an error if `fetch()` has not been called yet.
        """
        if self.data is None:
            raise ValueError("No data fetched yet. Use .fetch() first.")
        return self.data

    def save_to_csv(self, path: str = "data/raw/stock_data.csv"):
        """
        Save the fetched data to a CSV file.

        Parameters:
        - path: output file path
        """
        if self.data is None:
            raise ValueError("No data to save. Run fetch() first.")
        self.data.to_csv(path, index=False)
        print(f"Data saved to {path}")

    @staticmethod
    def load_from_csv(path: str) -> pd.DataFrame:
        """
        Load data from a previously saved CSV file.

        Performs:
        - Date parsing
        - Duplicate removal based on Date and Ticker
        - Sorting by Ticker and Date

        Parameters:
        - path: path to the CSV file

        Returns:
        - Cleaned DataFrame
        """
        try:
            df = pd.read_csv(path, parse_dates=["Date"])  # Automatically parse date column
            print(f"Data loaded from {path}")
        except Exception as e:
            raise ValueError(f"Error loading CSV from {path}: {e}")

        df = (
            df.drop_duplicates(subset=["Date", "Ticker"])  # Drop duplicate entries
            .sort_values(["Ticker", "Date"])  # Sort for consistency
            .reset_index(drop=True)
        )
        return df

    def quick_summary(self):
        """
        Print a quick summary: start date, end date, and number of rows per ticker.
        """
        if self.data is None:
            raise ValueError("No data fetched yet.")
        summary = self.data.groupby("Ticker")["Date"].agg(["min", "max", "count"])  # Aggregate summary
        print("\nData summary:")
        print(summary)
