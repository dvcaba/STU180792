import pandas as pd  # For data manipulation using DataFrame

class TechnicalIndicators:
    """Class to compute technical indicators for stock price data."""

    def __init__(self, df: pd.DataFrame):
        """Initialize with a DataFrame and check for required 'Close' column."""
        self.df = df.copy()
        if "Close" not in self.df.columns:
            raise ValueError("DataFrame must contain 'Close' column.")

    def _check_required_columns(self, required):
        """Internal helper to ensure required columns exist in the DataFrame."""
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def add_rsi(self, window: int = 14):
        """
        Add Relative Strength Index (RSI) to the DataFrame.

        RSI measures the speed and change of price movements and is typically used to identify overbought or oversold conditions.
        """
        self._check_required_columns(["Close"])
        delta = self.df["Close"].diff()  # Price difference between consecutive rows
        gain = delta.clip(lower=0)  # Keep only positive gains
        loss = -delta.clip(upper=0)  # Keep only negative losses, make them positive

        # Calculate rolling average of gains and losses
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        # Compute Relative Strength (RS) and RSI
        rs = avg_gain / avg_loss
        self.df["RSI"] = 100 - (100 / (1 + rs))  # RSI formula

        return self

    def add_macd(self, short_window: int = 12, long_window: int = 26, signal_window: int = 9):
        """
        Add MACD (Moving Average Convergence Divergence) and related values.

        MACD = EMA(12) - EMA(26)
        Signal line = EMA(9) of MACD
        Histogram = MACD - Signal
        """
        self._check_required_columns(["Close"])

        # Compute short and long-term exponential moving averages
        ema_short = self.df["Close"].ewm(span=short_window, adjust=False).mean()
        ema_long = self.df["Close"].ewm(span=long_window, adjust=False).mean()

        # MACD line and signal line
        macd = ema_short - ema_long
        signal = macd.ewm(span=signal_window, adjust=False).mean()

        # Store indicators in DataFrame
        self.df["MACD"] = macd
        self.df["MACD_Signal"] = signal
        self.df["MACD_Hist"] = macd - signal  # Histogram = MACD - Signal

        return self

    def add_bollinger_bands(self, window: int = 20, num_std: float = 2.0):
        """
        Add Bollinger Bands to the DataFrame.

        Bollinger Bands consist of:
        - Middle Band: Moving average
        - Upper Band: Middle + num_std * standard deviation
        - Lower Band: Middle - num_std * standard deviation
        """
        self._check_required_columns(["Close"])

        # Calculate rolling mean and standard deviation
        sma = self.df["Close"].rolling(window=window).mean()
        std = self.df["Close"].rolling(window=window).std()

        # Compute bands
        self.df["BB_Middle"] = sma
        self.df["BB_Upper"] = sma + num_std * std
        self.df["BB_Lower"] = sma - num_std * std

        return self

    def dropna(self):
        """Drop rows with any NaN values caused by rolling calculations."""
        self.df.dropna(inplace=True)
        return self

    def get_data(self):
        """Return the DataFrame with the computed technical indicators."""
        return self.df
