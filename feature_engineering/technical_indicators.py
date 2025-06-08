import pandas as pd

class TechnicalIndicators:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if "Close" not in self.df.columns:
            raise ValueError("DataFrame must contain 'Close' column.")

    def _check_required_columns(self, required):
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def add_rsi(self, window: int = 14):
        self._check_required_columns(["Close"])
        delta = self.df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        self.df["RSI"] = 100 - (100 / (1 + rs))
        return self

    def add_macd(self, short_window: int = 12, long_window: int = 26, signal_window: int = 9):
        self._check_required_columns(["Close"])
        ema_short = self.df["Close"].ewm(span=short_window, adjust=False).mean()
        ema_long = self.df["Close"].ewm(span=long_window, adjust=False).mean()
        macd = ema_short - ema_long
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        self.df["MACD"] = macd
        self.df["MACD_Signal"] = signal
        self.df["MACD_Hist"] = macd - signal
        return self

    def add_bollinger_bands(self, window: int = 20, num_std: float = 2.0):
        self._check_required_columns(["Close"])
        sma = self.df["Close"].rolling(window=window).mean()
        std = self.df["Close"].rolling(window=window).std()
        self.df["BB_Middle"] = sma
        self.df["BB_Upper"] = sma + num_std * std
        self.df["BB_Lower"] = sma - num_std * std
        return self

    def dropna(self):
        self.df.dropna(inplace=True)
        return self

    def get_data(self):
        return self.df
