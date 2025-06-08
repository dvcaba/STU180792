import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Union, List

class StockDataFetcher:
    def __init__(self, ticker: Union[str, List[str]], start_date: str = "2020-01-01", end_date: str = None):
        """
        Inicializa el objeto para descargar datos históricos de acciones usando yfinance.
        """
        if isinstance(ticker, str):
            self.tickers = [ticker.upper()]
        elif isinstance(ticker, list):
            self.tickers = [t.upper() for t in ticker]
        else:
            raise ValueError("Ticker must be a string or list of strings.")

        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.today().strftime('%Y-%m-%d')
        self.data = None

    def fetch(self) -> pd.DataFrame:
        """
        Descarga los datos de todos los tickers y devuelve un DataFrame combinado.
        """
        all_data = []

        for ticker in self.tickers:
            print(f"Fetching: {ticker}")
            try:
                df = yf.download(
                    ticker,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    auto_adjust=True
                )

                if df.empty:
                    print(f"Warning: No data for {ticker}")
                    continue

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]

                df = df.reset_index()
                df["Ticker"] = ticker
                all_data.append(df)

            except Exception as e:
                print(f"Error fetching {ticker}: {e}")

        if not all_data:
            raise ValueError("No data retrieved for any tickers.")

        self.data = pd.concat(all_data, ignore_index=True)
        return self.data

    def get_data(self) -> pd.DataFrame:
        """
        Devuelve los datos descargados. Lanza error si aún no se ha llamado a fetch().
        """
        if self.data is None:
            raise ValueError("No data fetched yet. Use .fetch() first.")
        return self.data

    def save_to_csv(self, path: str = "data/raw/stock_data.csv"):
        """
        Guarda los datos descargados en un archivo CSV.
        """
        if self.data is None:
            raise ValueError("No data to save. Run fetch() first.")
        self.data.to_csv(path, index=False)
        print(f"Data saved to {path}")

    @staticmethod
    def load_from_csv(path: str) -> pd.DataFrame:
        """
        Carga datos desde un archivo CSV previamente guardado.
        """
        try:
            df = pd.read_csv(path, parse_dates=["Date"])
            print(f"Data loaded from {path}")
            return df
        except Exception as e:
            raise ValueError(f"Error loading CSV from {path}: {e}")

    def quick_summary(self):
        """
        Muestra un resumen de fechas y cantidad de datos por ticker.
        """
        if self.data is None:
            raise ValueError("No data fetched yet.")
        summary = self.data.groupby("Ticker")["Date"].agg(["min", "max", "count"])
        print("\nData summary:")
        print(summary)
