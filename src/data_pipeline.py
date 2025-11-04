import os
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
import pickle
from darts import TimeSeries

# --- Configuration ---
TICKER = "AAPL"
START_DATE = "2015-01-01"
END_DATE = None  # Fetches data up to the current day
VAL_SIZE_DAYS = 252 * 2  # ~2 years of trading days for a robust validation set

# Define paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / f"{TICKER}_historical.csv"
TRAIN_SERIES_PATH = PROJECT_ROOT / "data" / "processed" / f"{TICKER}_TS_train.pkl"
VAL_SERIES_PATH = PROJECT_ROOT / "data" / "processed" / f"{TICKER}_TS_val.pkl"


def fetch_and_save_raw_data(
        ticker: str = TICKER, start_date: str = START_DATE, end_date: Optional[str] = END_DATE
) -> pd.DataFrame:
        """Fetch historical stock data using yfinance and save to the raw data folder.

        Returns an empty DataFrame on failure.
        """
        print(f"-> Fetching raw data for {ticker} from {start_date} to today...")

        # Try download
        try:
                df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        except Exception as exc:
                print(f"!!! CRITICAL DOWNLOAD ERROR: {exc}")
                df = pd.DataFrame()

        # If download failed or returned empty, attempt to load an existing file
        if df.empty:
                print("!!! WARNING: Data download returned empty. Trying to load previously saved raw data...")
                if RAW_DATA_PATH.exists():
                        try:
                                df = pd.read_csv(RAW_DATA_PATH, index_col=0, parse_dates=True)
                                print("-> Successfully loaded previous raw data from disk.")
                        except Exception as exc:
                                print(f"!!! FAILED to load previous raw data: {exc}")
                                return pd.DataFrame()
                else:
                        print("!!! No previous raw data found on disk. Exiting fetch with empty DataFrame.")
                        return pd.DataFrame()

        # Ensure index is a DatetimeIndex
        if not df.empty and not isinstance(df.index, pd.DatetimeIndex):
                try:
                        df.index = pd.to_datetime(df.index)
                except Exception:
                        # If conversion fails, return empty df to avoid downstream crashes
                        print("!!! ERROR: Failed to convert index to DatetimeIndex.")
                        return pd.DataFrame()

        # Ensure raw data directory exists and save
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not df.empty:
                try:
                        df.to_csv(RAW_DATA_PATH)
                        print(f"-> Raw data saved to: {RAW_DATA_PATH}")
                except Exception as exc:
                        print(f"!!! WARNING: Failed to save raw data to disk: {exc}")

        return df


def preprocess_to_darts_series(df: pd.DataFrame, target_col: str = "Close") -> Optional[TimeSeries]:
        """Clean the DataFrame and convert the `target_col` into a Darts TimeSeries.

        Returns None on failure.
        """
        if df is None or df.empty:
                print("!!! Preprocessing skipped: Input DataFrame is empty or None.")
                return None

        if target_col not in df.columns:
                print(f"!!! ERROR: Target column '{target_col}' not found. Available columns: {df.columns.tolist()}")
                return None

        series_df = df[[target_col]].copy()

        # Fill missing values if any
        if series_df.isnull().values.any():
                print("-> Warning: Filling NaN values using forward-fill method.")
                series_df = series_df.fillna(method="ffill")

        # Create TimeSeries. Use value_cols to be explicit.
        try:
                series = TimeSeries.from_dataframe(series_df, value_cols=target_col, freq="B")
        except Exception as exc:
                print(f"!!! ERROR creating TimeSeries: {exc}")
                return None

        print(f"-> Created TimeSeries of length: {len(series)}")
        return series


def split_and_save_series(series: TimeSeries, val_size: int = VAL_SIZE_DAYS):
        """Split the series into training and validation sets and save them to disk.

        Returns (train_series, val_series) or (None, None) on failure.
        """
        if series is None:
                print("!!! Skipping split and save: TimeSeries object is None.")
                return None, None

        if val_size <= 0:
                print("!!! ERROR: val_size must be > 0.")
                return None, None

        if len(series) < val_size:
                print("!!! ERROR: Series is too short to create the requested validation set size.")
                return None, None

        train_series, val_series = series[:-val_size], series[-val_size:]

        # Ensure the processed directory exists
        TRAIN_SERIES_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Save with the highest available protocol
        for s, path in [(train_series, TRAIN_SERIES_PATH), (val_series, VAL_SERIES_PATH)]:
                try:
                        with open(path, "wb") as f:
                                pickle.dump(s, f, protocol=pickle.HIGHEST_PROTOCOL)
                        print(f"-> Series saved to: {path}")
                except Exception as exc:
                        print(f"!!! WARNING: Failed to save series to {path}: {exc}")

        return train_series, val_series


def main():
        """Execute the full data pipeline: fetch -> preprocess -> split -> save."""
        raw_df = fetch_and_save_raw_data()

        series = preprocess_to_darts_series(raw_df)

        if series is not None:
                split_and_save_series(series)
        else:
                print("\nPipeline failed due to errors. Cannot proceed to model training.")


if __name__ == "__main__":
        main()