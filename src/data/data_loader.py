""" 
Data loading utilities for the multi-strategy fund codebase.

This module provides functions and classes to load and preprocess financial data.
It supports loading data from various sources, handling missing, duplicated entries for robust data ingestion.

Dependencises:
    - pandas
    - numpy
    - yfinance
"""
import pandas as pd
import numpy as np
import yfinance as yf

class DataLoader:
    def __init__(self, tickers, start_date, end_date, interval='1d'):
        """
        Initializes the DataLoader with specified parameters.

        Args:
            tickers (list): List of ticker symbols to download data for.
            start_date (str): Start date for data in 'YYYY-MM-DD' format.
            end_date (str): End date for data in 'YYYY-MM-DD' format.
            interval (str): Data interval (e.g., '1d', '1h', '5m').
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval

    def load_data(self):
        """
        Downloads and preprocesses data for the specified tickers.

        Returns:
            pd.DataFrame: A DataFrame containing the combined data for all tickers.
        """
        all_data = []
        for ticker in self.tickers:
            data = yf.download(ticker, start=self.start_date, end=self.end_date, interval=self.interval)
            if data is not None and not data.empty:
                data['Ticker'] = ticker
                all_data.append(data)

        combined_data = pd.concat(all_data)
        combined_data.reset_index(inplace=True)

        # Handle missing values by forward filling
        combined_data.ffill( inplace=True)
        combined_data.bfill( inplace=True)

        # Remove duplicates
        combined_data.drop_duplicates(inplace=True)

        return combined_data

    def get_data_for_ticker(self, ticker):
        """
        Retrieves data for a specific ticker.

        Args:
            ticker (str): The ticker symbol to retrieve data for.

        Returns:
            pd.DataFrame: A DataFrame containing the data for the specified ticker.
        """
        if ticker in self.tickers:
            return self.load_data().loc[self.load_data()['Ticker'] == ticker]
        else:
            return pd.DataFrame()
        
    def ariticdb_load(self, db_path):
        """
        Loads data from an AriticDB database.

        Args:
            db_path (str): Path to the AriticDB database file.
        """
        # Implement the logic to load data from the AriticDB database
        pass

        

def main():
    # Example usage
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    data_loader = DataLoader(tickers, '2022-01-01', '2022-12-31')
    data = data_loader.load_data()
    print(data.head())

if __name__ == "__main__":
    main()
