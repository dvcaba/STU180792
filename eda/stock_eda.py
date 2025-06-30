import os  # For file and directory operations
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For advanced visualization (heatmaps, distributions, etc.)

class StockEDA:
    """Utility class for basic exploratory analysis of stock data."""

    def __init__(self, data: pd.DataFrame):
        """Initialize with a dataframe containing stock prices."""
        self.data = data.copy()  # Make a copy to avoid modifying original

        # Check that required columns are present
        expected_cols = {"Date", "Open", "High", "Low", "Close", "Volume", "Ticker"}
        if not expected_cols.issubset(set(self.data.columns)):
            raise ValueError(f"Missing expected columns in data: {expected_cols - set(self.data.columns)}")

        # Ensure 'Date' is datetime type and sort data
        self.data["Date"] = pd.to_datetime(self.data["Date"])
        self.data.sort_values(by=["Ticker", "Date"], inplace=True)

        # Calculate daily return as percent change of 'Close' price
        self.data["Daily Return"] = self.data.groupby("Ticker")["Close"].pct_change()

    def show_basic_info(self):
        """Print shape, info, and missing value counts."""
        print("Shape:", self.data.shape)
        print("\nInfo:")
        print(self.data.info())
        print("\nMissing values:")
        print(self.data.isnull().sum())

    def show_summary_stats(self):
        """Print statistical summary of the dataframe."""
        print("\nStatistical Summary:")
        print(self.data.describe())

    def _save_or_show(self, save_dir, filename):
        """Helper function to save the plot or show it depending on save_dir."""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, filename))
            plt.close()
        else:
            plt.show()

    def plot_price_trend(self, save_dir=None):
        """Plot closing price over time for each ticker."""
        plt.figure(figsize=(12, 6))
        for ticker in self.data["Ticker"].unique():
            subset = self.data[self.data["Ticker"] == ticker]
            plt.plot(subset["Date"], subset["Close"], label=ticker)
        plt.title("Closing Price Over Time")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()
        plt.grid(True)
        self._save_or_show(save_dir, "price_trend.png")

    def plot_volume(self, save_dir=None):
        """Plot trading volume over time for each ticker."""
        plt.figure(figsize=(12, 6))
        for ticker in self.data["Ticker"].unique():
            subset = self.data[self.data["Ticker"] == ticker]
            plt.plot(subset["Date"], subset["Volume"], label=ticker)
        plt.title("Trading Volume Over Time")
        plt.xlabel("Date")
        plt.ylabel("Volume")
        plt.legend()
        plt.grid(True)
        self._save_or_show(save_dir, "volume_trend.png")

    def plot_daily_return_distribution(self, save_dir=None):
        """Plot kernel density estimate of daily return distribution per ticker."""
        plt.figure(figsize=(10, 6))
        for ticker in self.data["Ticker"].unique():
            subset = self.data[self.data["Ticker"] == ticker]
            sns.kdeplot(subset["Daily Return"].dropna(), label=ticker, fill=True)
        plt.title("Distribution of Daily Returns")
        plt.xlabel("Daily Return")
        plt.legend()
        plt.grid(True)
        self._save_or_show(save_dir, "daily_return_distribution.png")

    def plot_correlation_matrix(self, save_dir=None):
        """Plot heatmap of correlation between tickers' daily returns."""
        pivoted = self.data.pivot(index="Date", columns="Ticker", values="Close")
        corr = pivoted.pct_change().corr()
        if corr.empty:
            print("Warning: correlation matrix is empty, skipping heatmap.")
            return
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Correlation of Daily Returns")
        self._save_or_show(save_dir, "correlation_matrix.png")

    def plot_outliers(self, column="Daily Return", save_dir=None):
        """Plot boxplot to show outliers of a column grouped by ticker."""
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=self.data, x="Ticker", y=column)
        plt.title(f"Outliers in {column} by Ticker")
        plt.grid(True)
        self._save_or_show(save_dir, f"outliers_{column}.png")

    def plot_rolling_volatility(self, window=20, save_dir=None):
        """Plot rolling standard deviation (volatility) over time."""
        plt.figure(figsize=(12, 6))
        for ticker in self.data["Ticker"].unique():
            subset = self.data[self.data["Ticker"] == ticker].copy()
            subset["Volatility"] = subset["Daily Return"].rolling(window).std()
            plt.plot(subset["Date"], subset["Volatility"], label=f"{ticker}")
        plt.title(f"{window}-Day Rolling Volatility")
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.legend()
        plt.grid(True)
        self._save_or_show(save_dir, f"rolling_volatility_{window}d.png")

    def plot_cumulative_returns(self, save_dir=None):
        """Plot cumulative returns for each ticker."""
        plt.figure(figsize=(12, 6))
        for ticker in self.data["Ticker"].unique():
            subset = self.data[self.data["Ticker"] == ticker].copy()
            subset["Cumulative Return"] = (1 + subset["Daily Return"]).cumprod()
            plt.plot(subset["Date"], subset["Cumulative Return"], label=ticker)
        plt.title("Cumulative Returns Over Time")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True)
        self._save_or_show(save_dir, "cumulative_returns.png")

    def run_all(self, save_dir=None):
        """Run all EDA functions and generate all plots."""
        self.show_basic_info()
        self.show_summary_stats()
        self.plot_price_trend(save_dir)
        self.plot_volume(save_dir)
        self.plot_daily_return_distribution(save_dir)
        self.plot_correlation_matrix(save_dir)
        self.plot_outliers(save_dir=save_dir)
        self.plot_rolling_volatility(window=20, save_dir=save_dir)
        self.plot_cumulative_returns(save_dir)
