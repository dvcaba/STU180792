import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Union

class DataPreprocessor:
    def __init__(
        self,
        data: pd.DataFrame,
        scale: Union[str, bool] = "standard",  # "standard", "minmax", o False
        model_type: str = "linear",
        target_type: str = "price",            # "price", "return", o "direction"
        forecast_horizon: int = 1
    ):
        self.raw_data = data.copy()
        self.processed_data = None
        self.scaler = None
        self.scale = scale
        self.model_type = model_type
        self.target_type = target_type
        self.forecast_horizon = forecast_horizon

    def preprocess(self) -> pd.DataFrame:
        df = self.raw_data.copy()

        # === Target ===
        if self.target_type == "price":
            df["Target"] = df.groupby("Ticker")["Close"].shift(-self.forecast_horizon)

        elif self.target_type == "return":
            df["Target"] = df.groupby("Ticker")["Close"].pct_change().shift(-self.forecast_horizon)

        elif self.target_type == "direction":
            df["Future_Return"] = df.groupby("Ticker")["Close"].pct_change().shift(-self.forecast_horizon)
            df["Target"] = (df["Future_Return"] > 0).astype(int)
            df.drop(columns=["Future_Return"], inplace=True)

        else:
            raise ValueError("target_type must be 'price', 'return' or 'direction'.")

        # === Features base ===
        df["Daily Return"] = df.groupby("Ticker")["Close"].pct_change()
        df["Volatility_5d"] = df.groupby("Ticker")["Close"].transform(lambda x: x.pct_change().rolling(5).std())
        df["MA_5"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(window=5).mean())
        df["MA_10"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(window=10).mean())

        # === Lags de precio ===
        for i in range(1, 11):
            df[f"Lag_{i}"] = df.groupby("Ticker")["Close"].shift(i)

        # === Lags de retorno ===
        for i in range(1, 6):
            df[f"Return_Lag_{i}"] = df.groupby("Ticker")["Daily Return"].shift(i)

        # === Diferencias intradía ===
        df["Delta_Close"] = df["Close"] - df["Open"]
        df["Range"] = df["High"] - df["Low"]

        # === Indicadores técnicos si existen ===
        technicals = [
            "RSI", "MACD", "MACD_Signal", "MACD_Hist",
            "BB_Middle", "BB_Upper", "BB_Lower"
        ]
        tech_features = [col for col in technicals if col in df.columns]

        # === Asegurar tipo datetime ===
        df["Date"] = pd.to_datetime(df["Date"])

        # === Nuevas features añadidas ===
        df["Weekday"] = df["Date"].dt.weekday
        df["Volume_Change"] = df.groupby("Ticker")["Volume"].pct_change()
        df["Price_to_Volume"] = df["Close"] / (df["Volume"] + 1e-6)
        df["Momentum_3d"] = df.groupby("Ticker")["Close"].pct_change(periods=3)
        df["Rolling_Return_5d"] = df.groupby("Ticker")["Close"].pct_change(periods=5)

        # === Limpiar nulos y valores infinitos ===
        df.replace([np.inf, -np.inf], pd.NA, inplace=True)
        df.dropna(inplace=True)

        # === Construcción de features ===
        base_features = [
            "Open", "High", "Low", "Close", "Volume",
            "Daily Return", "Volatility_5d", "MA_5", "MA_10",
            "Delta_Close", "Range", "Weekday",
            "Volume_Change", "Price_to_Volume",
            "Momentum_3d", "Rolling_Return_5d"
        ]
        lag_features = [f"Lag_{i}" for i in range(1, 11)]
        return_features = [f"Return_Lag_{i}" for i in range(1, 6)]
        features = base_features + lag_features + return_features + tech_features

        all_columns = ["Date", "Ticker"] + features + ["Target"]
        df_features = df[all_columns].copy()

        # === Escalado (opcional) ===
        if self.scale and self.model_type != "xgboost":
            if self.scale == "standard":
                self.scaler = StandardScaler()
            elif self.scale == "minmax":
                self.scaler = MinMaxScaler()
            else:
                raise ValueError("Invalid scale option. Use 'standard', 'minmax' or False.")

            df_features[features] = self.scaler.fit_transform(df_features[features])

        self.processed_data = df_features
        return df_features
