import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class StockEDA:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

        # Validate structure
        expected_cols = {"Date", "Open", "High", "Low", "Close", "Volume", "Ticker"}
        if not expected_cols.issubset(set(self.data.columns)):
            raise ValueError(f"Missing expected columns in data: {expected_cols - set(self.data.columns)}")

        self.data["Date"] = pd.to_datetime(self.data["Date"])
        self.data.sort_values(by=["Ticker", "Date"], inplace=True)
        self.data["Daily Return"] = self.data.groupby("Ticker")["Close"].pct_change()

    def show_basic_info(self):
        print("Shape:", self.data.shape)
        print("\nInfo:")
        print(self.data.info())
        print("\nMissing values:")
        print(self.data.isnull().sum())

    def show_summary_stats(self):
        print("\nStatistical Summary:")
        print(self.data.describe())

    def _save_or_show(self, save_dir, filename):
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, filename))
            plt.close()
        else:
            plt.show()

    def plot_price_trend(self, save_dir=None):
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
        pivoted = self.data.pivot(index="Date", columns="Ticker", values="Close")
        corr = pivoted.pct_change().corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Correlation of Daily Returns")
        self._save_or_show(save_dir, "correlation_matrix.png")

    def plot_outliers(self, column="Daily Return", save_dir=None):
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=self.data, x="Ticker", y=column)
        plt.title(f"Outliers in {column} by Ticker")
        plt.grid(True)
        self._save_or_show(save_dir, f"outliers_{column}.png")

    def plot_rolling_volatility(self, window=20, save_dir=None):
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
        self.show_basic_info()
        self.show_summary_stats()
        self.plot_price_trend(save_dir)
        self.plot_volume(save_dir)
        self.plot_daily_return_distribution(save_dir)
        self.plot_correlation_matrix(save_dir)
        self.plot_outliers(save_dir=save_dir)
        self.plot_rolling_volatility(window=20, save_dir=save_dir)
        self.plot_cumulative_returns(save_dir)
