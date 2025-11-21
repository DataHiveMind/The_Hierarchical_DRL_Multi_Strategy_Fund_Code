"""
Data loading utilities for the multi-strategy fund codebase.

This module provides functions and classes to load and preprocess financial data.
It supports loading data from various sources, handling missing, duplicated entries for robust data ingestion.

Dependencises:
    - pandas
    - numpy
    - yfinance
"""

import arcticdb as adb
import pandas as pd
import yfinance as yf


class DataLoader:
    def __init__(
        self,
        tickers,
        start_date,
        end_date,
        interval="1d",
        db_url="lmdb://financial_data",
    ):
        """
        Initializes the DataLoader with specified parameters.

        Args:
            tickers (list): List of ticker symbols to download data for.
            start_date (str): Start date for data in 'YYYY-MM-DD' format.
            end_date (str): End date for data in 'YYYY-MM-DD' format.
            interval (str): Data interval (e.g., '1d', '1h', '5m').
            db_url (str): ArcticDB connection URI (default: 'lmdb://financial_data' for local LMDB storage).
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.arctic_database = adb.Arctic(db_url)

    def load_raw_data(self):
        """
        Downloads and preprocesses data for the specified tickers.

        Returns:
            pd.DataFrame: A DataFrame containing the combined data for all tickers.
        """
        data = yf.download(
            tickers=self.tickers,
            start=self.start_date,
            end=self.end_date,
            interval=self.interval,
            auto_adjust=False,
        )

        data.reset_index(inplace=True)

        # Handle missing values by forward filling
        data.ffill(inplace=True)
        data.bfill(inplace=True)

        # Remove duplicates
        data.drop_duplicates(inplace=True)

        return data

    def get_data_for_ticker(self, ticker: str) -> pd.DataFrame:
        """
        Retrieves data for a specific ticker.

        Args:
            ticker (str): The ticker symbol to retrieve data for.

        Returns:
            pd.DataFrame: A DataFrame containing the data for the specified ticker.
        """
        if ticker in self.tickers:
            return self.load_data().loc[self.load_data()["Ticker"] == ticker]
        else:
            return pd.DataFrame()

    def store_data(self, collection_name: str, df: pd.DataFrame):
        """Store the cleaned data into ArcticDB."""
        lib = self.arctic_database.get_library(collection_name, create_if_missing=True)
        lib.write("data", df)

    def read_data(self, collection_name: str, symbol: str = "data") -> pd.DataFrame:
        """Read data from ArcticDB collection."""
        lib = self.arctic_database.get_library(collection_name, create_if_missing=False)
        df = lib.read(symbol).data
        return df

    def display_data(self, collection_name: str, symbol: str = "data", head: int = 10):
        """Display data from ArcticDB collection with summary statistics."""
        try:
            df = self.read_data(collection_name, symbol)

            print(f"\n{'=' * 60}")
            print(f"Data from collection: '{collection_name}' (symbol: '{symbol}')")
            print(f"{'=' * 60}\n")

            print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
            print(f"Date range: {df.index.min()} to {df.index.max()}\n")

            print(f"First {head} rows:")
            print(df.head(head))

            print(f"\nLast {head} rows:")
            print(df.tail(head))

            print("\nSummary Statistics:")
            print(df.describe())

            print("\nColumn Names:")
            print(df.columns.tolist())

            print("\nData Types:")
            print(df.dtypes)

            print(f"\n{'=' * 60}\n")

        except Exception as e:
            print(f"Error reading data from collection '{collection_name}': {e}")


def main():
    # Example usage
    tickers = ["AAPL", "MSFT", "GOOGL"]
    data_loader = DataLoader(tickers, "2022-01-01", "2022-12-31")
    data = data_loader.load_raw_data()
    print(data.head())

    # Store and display data from ArcticDB
    data_loader.store_data("financial_data", data)
    data_loader.display_data("financial_data")


if __name__ == "__main__":
    main()
