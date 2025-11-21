"""
Feature engineering utilities for multi-strategy quantitative trading.

This module provides comprehensive feature engineering capabilities for various
trading strategies including technical indicators, statistical features, volatility
metrics, and market microstructure signals.

Dependencies:
    - pandas
    - numpy
    - scipy
    - ta (technical analysis library)
"""

import numpy as np
import pandas as pd
from typing import List, Dict
import warnings

warnings.filterwarnings("ignore")


class FeatureEngineer:
    """
    Main feature engineering class for creating trading signals and features
    across multiple asset classes and strategies.

    Note: This class works with single-ticker data. For multi-ticker DataFrames,
    extract individual ticker data first or use MultiAssetFeatureEngineer.
    """

    def __init__(self, df: pd.DataFrame, price_col: str = "Close"):
        """
        Initialize the FeatureEngineer with market data.

        Args:
            df (pd.DataFrame): Input dataframe with OHLCV data (single ticker)
            price_col (str): Column name for price (default: 'Close')
        """
        self.df = df.copy()
        self.price_col = price_col

        # Validate that we're working with single-column price data
        if isinstance(self.df.get(self.price_col), pd.DataFrame):
            raise ValueError(
                f"The '{self.price_col}' column contains multiple columns (multi-ticker data). "
                "Please provide data for a single ticker or extract individual ticker data first."
            )

    def add_returns(self, periods: List[int] = [1, 5, 10, 20, 60]) -> pd.DataFrame:
        """
        Calculate returns over multiple periods.

        Args:
            periods (List[int]): List of periods for return calculation

        Returns:
            pd.DataFrame: DataFrame with return columns added
        """
        for period in periods:
            self.df[f"return_{period}d"] = self.df[self.price_col].pct_change(period)
            self.df[f"log_return_{period}d"] = np.log(
                self.df[self.price_col] / self.df[self.price_col].shift(period)
            )
        return self.df

    def add_moving_averages(
        self, windows: List[int] = [5, 10, 20, 50, 200]
    ) -> pd.DataFrame:
        """
        Calculate simple and exponential moving averages.

        Args:
            windows (List[int]): List of window sizes

        Returns:
            pd.DataFrame: DataFrame with MA columns added
        """
        for window in windows:
            self.df[f"sma_{window}"] = (
                self.df[self.price_col].rolling(window=window).mean()
            )
            self.df[f"ema_{window}"] = (
                self.df[self.price_col].ewm(span=window, adjust=False).mean()
            )

            # Price relative to moving averages
            self.df[f"price_to_sma_{window}"] = (
                self.df[self.price_col] / self.df[f"sma_{window}"] - 1
            )
            self.df[f"price_to_ema_{window}"] = (
                self.df[self.price_col] / self.df[f"ema_{window}"] - 1
            )

        return self.df

    def add_volatility_features(
        self, windows: List[int] = [5, 10, 20, 60]
    ) -> pd.DataFrame:
        """
        Calculate various volatility measures.

        Args:
            windows (List[int]): List of window sizes for volatility calculation

        Returns:
            pd.DataFrame: DataFrame with volatility features added
        """
        returns = self.df[self.price_col].pct_change()

        for window in windows:
            # Historical volatility (annualized)
            self.df[f"volatility_{window}d"] = returns.rolling(
                window=window
            ).std() * np.sqrt(252)

            # Parkinson volatility (uses high-low range)
            if "High" in self.df.columns and "Low" in self.df.columns:
                hl_ratio = np.log(self.df["High"] / self.df["Low"])
                self.df[f"parkinson_vol_{window}d"] = np.sqrt(
                    hl_ratio.rolling(window=window).apply(
                        lambda x: (1 / (4 * np.log(2))) * np.mean(x**2)
                    )
                    * 252
                )

            # Realized volatility (sum of squared returns)
            self.df[f"realized_vol_{window}d"] = np.sqrt(
                (returns**2).rolling(window=window).sum() * 252
            )

            # Volatility of volatility
            self.df[f"vol_of_vol_{window}d"] = (
                self.df[f"volatility_{window}d"].rolling(window=window).std()
            )

        return self.df

    def add_momentum_indicators(self) -> pd.DataFrame:
        """
        Add momentum-based technical indicators.

        Returns:
            pd.DataFrame: DataFrame with momentum indicators added
        """
        # RSI (Relative Strength Index)
        for period in [14, 28]:
            delta = self.df[self.price_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            self.df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        # MACD (Moving Average Convergence Divergence)
        exp1 = self.df[self.price_col].ewm(span=12, adjust=False).mean()
        exp2 = self.df[self.price_col].ewm(span=26, adjust=False).mean()
        self.df["macd"] = exp1 - exp2
        self.df["macd_signal"] = self.df["macd"].ewm(span=9, adjust=False).mean()
        self.df["macd_histogram"] = self.df["macd"] - self.df["macd_signal"]

        # Rate of Change (ROC)
        for period in [10, 20, 50]:
            self.df[f"roc_{period}"] = (
                (self.df[self.price_col] - self.df[self.price_col].shift(period))
                / self.df[self.price_col].shift(period)
                * 100
            )

        return self.df

    def add_bollinger_bands(
        self, window: int = 20, num_std: float = 2.0
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands and related features.

        Args:
            window (int): Rolling window size
            num_std (float): Number of standard deviations

        Returns:
            pd.DataFrame: DataFrame with Bollinger Band features added
        """
        sma = self.df[self.price_col].rolling(window=window).mean()
        std = self.df[self.price_col].rolling(window=window).std()

        self.df[f"bb_upper_{window}"] = sma + (std * num_std)
        self.df[f"bb_middle_{window}"] = sma
        self.df[f"bb_lower_{window}"] = sma - (std * num_std)

        # Bollinger Band Width
        self.df[f"bb_width_{window}"] = (
            self.df[f"bb_upper_{window}"] - self.df[f"bb_lower_{window}"]
        ) / self.df[f"bb_middle_{window}"]

        # Price position within bands
        self.df[f"bb_position_{window}"] = (
            self.df[self.price_col] - self.df[f"bb_lower_{window}"]
        ) / (self.df[f"bb_upper_{window}"] - self.df[f"bb_lower_{window}"])

        return self.df

    def add_volume_features(self) -> pd.DataFrame:
        """
        Add volume-based features (if volume data available).

        Returns:
            pd.DataFrame: DataFrame with volume features added
        """
        if "Volume" not in self.df.columns:
            return self.df

        # Volume moving averages
        for window in [5, 10, 20]:
            self.df[f"volume_sma_{window}"] = (
                self.df["Volume"].rolling(window=window).mean()
            )
            self.df[f"volume_ratio_{window}"] = (
                self.df["Volume"] / self.df[f"volume_sma_{window}"]
            )

        # On-Balance Volume (OBV)
        self.df["obv"] = (
            (np.sign(self.df[self.price_col].diff()) * self.df["Volume"])
            .fillna(0)
            .cumsum()
        )

        # Volume-Weighted Average Price (VWAP) - daily reset
        if "High" in self.df.columns and "Low" in self.df.columns:
            typical_price = (
                self.df["High"] + self.df["Low"] + self.df[self.price_col]
            ) / 3
            self.df["vwap"] = (typical_price * self.df["Volume"]).cumsum() / self.df[
                "Volume"
            ].cumsum()
            self.df["price_to_vwap"] = self.df[self.price_col] / self.df["vwap"] - 1

        # Money Flow Index (MFI)
        if "High" in self.df.columns and "Low" in self.df.columns:
            typical_price = (
                self.df["High"] + self.df["Low"] + self.df[self.price_col]
            ) / 3
            money_flow = typical_price * self.df["Volume"]

            positive_flow = (
                money_flow.where(typical_price > typical_price.shift(1), 0)
                .rolling(14)
                .sum()
            )
            negative_flow = (
                money_flow.where(typical_price < typical_price.shift(1), 0)
                .rolling(14)
                .sum()
            )

            mfi = 100 - (100 / (1 + positive_flow / negative_flow))
            self.df["mfi_14"] = mfi

        return self.df

    def add_statistical_features(self, windows: List[int] = [20, 60]) -> pd.DataFrame:
        """
        Add statistical features for mean reversion and trend detection.

        Args:
            windows (List[int]): List of window sizes

        Returns:
            pd.DataFrame: DataFrame with statistical features added
        """
        returns = self.df[self.price_col].pct_change()

        for window in windows:
            # Z-score (for mean reversion)
            rolling_mean = self.df[self.price_col].rolling(window=window).mean()
            rolling_std = self.df[self.price_col].rolling(window=window).std()
            self.df[f"zscore_{window}"] = (
                self.df[self.price_col] - rolling_mean
            ) / rolling_std

            # Skewness and Kurtosis
            self.df[f"skewness_{window}"] = returns.rolling(window=window).skew()
            self.df[f"kurtosis_{window}"] = returns.rolling(window=window).kurt()

            # Autocorrelation
            self.df[f"autocorr_{window}"] = returns.rolling(window=window).apply(
                lambda x: x.autocorr(), raw=False
            )

            # Rolling Sharpe Ratio (assuming risk-free rate = 0)
            rolling_return = returns.rolling(window=window).mean()
            rolling_vol = returns.rolling(window=window).std()
            self.df[f"sharpe_{window}"] = (rolling_return / rolling_vol) * np.sqrt(252)

        return self.df

    def add_trend_features(self) -> pd.DataFrame:
        """
        Add trend detection and strength features.

        Returns:
            pd.DataFrame: DataFrame with trend features added
        """
        # ADX (Average Directional Index)
        if "High" in self.df.columns and "Low" in self.df.columns:
            high = self.df["High"]
            low = self.df["Low"]
            close = self.df[self.price_col]

            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()

            self.df["atr_14"] = atr

            # Directional Movement
            up_move = high - high.shift()
            down_move = low.shift() - low

            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

            plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / atr
            minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / atr

            self.df["plus_di"] = plus_di
            self.df["minus_di"] = minus_di

            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            self.df["adx"] = dx.rolling(14).mean()

        # Linear regression slope (trend strength)
        for window in [10, 20, 50]:
            self.df[f"slope_{window}"] = (
                self.df[self.price_col]
                .rolling(window=window)
                .apply(
                    lambda x: np.polyfit(np.arange(len(x)), x, 1)[0]
                    if len(x) == window
                    else np.nan,
                    raw=True,
                )
            )

        return self.df

    def add_spread_features(
        self, df_pair: pd.DataFrame, price_col_pair: str = "Close"
    ) -> pd.DataFrame:
        """
        Add spread features for pairs trading / statistical arbitrage.

        Args:
            df_pair (pd.DataFrame): Price data for the paired asset
            price_col_pair (str): Column name for pair price

        Returns:
            pd.DataFrame: DataFrame with spread features added
        """
        # Price ratio
        self.df["price_ratio"] = self.df[self.price_col] / df_pair[price_col_pair]

        # Log price ratio
        self.df["log_price_ratio"] = np.log(self.df[self.price_col]) - np.log(
            df_pair[price_col_pair]
        )

        # Spread z-score
        for window in [20, 60]:
            spread_mean = self.df["log_price_ratio"].rolling(window=window).mean()
            spread_std = self.df["log_price_ratio"].rolling(window=window).std()
            self.df[f"spread_zscore_{window}"] = (
                self.df["log_price_ratio"] - spread_mean
            ) / spread_std

        # Half-life of mean reversion (Ornstein-Uhlenbeck process)
        spread = self.df["log_price_ratio"].dropna()
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()

        # Align indices
        spread_lag = spread_lag[spread_diff.index]

        if len(spread_diff) > 1:
            regression = np.polyfit(spread_lag, spread_diff, 1)
            half_life = -np.log(2) / regression[0] if regression[0] < 0 else np.nan
            self.df["spread_half_life"] = half_life

        return self.df

    def add_market_regime_features(self) -> pd.DataFrame:
        """
        Add features for market regime detection.

        Returns:
            pd.DataFrame: DataFrame with regime features added
        """
        returns = self.df[self.price_col].pct_change()

        # Bull/Bear market indicator (based on MA crossover)
        self.df["regime_ma"] = np.where(
            self.df[self.price_col] > self.df[self.price_col].rolling(200).mean(),
            1,  # Bull
            -1,  # Bear
        )

        # Volatility regime (high/low volatility)
        vol_20 = returns.rolling(20).std()
        vol_median = vol_20.rolling(252).median()
        self.df["regime_vol"] = np.where(
            vol_20 > vol_median, 1, -1
        )  # 1=High vol, -1=Low vol

        # Trend strength regime
        if "adx" in self.df.columns:
            self.df["regime_trend"] = np.where(
                self.df["adx"] > 25,
                1,  # Trending
                0,  # Ranging
            )

        return self.df

    def add_microstructure_features(self) -> pd.DataFrame:
        """
        Add market microstructure features (for market making strategy).

        Returns:
            pd.DataFrame: DataFrame with microstructure features added
        """
        if "High" not in self.df.columns or "Low" not in self.df.columns:
            return self.df

        # Bid-ask spread proxy (high-low range)
        self.df["hl_spread"] = self.df["High"] - self.df["Low"]
        self.df["hl_spread_pct"] = self.df["hl_spread"] / self.df[self.price_col]

        # Roll's spread estimator
        price_changes = self.df[self.price_col].diff()
        covariance = price_changes.rolling(20).cov(price_changes.shift(1))
        self.df["roll_spread"] = 2 * np.sqrt(-covariance.clip(upper=0))

        # Amihud illiquidity measure
        if "Volume" in self.df.columns:
            self.df["amihud_illiquidity"] = abs(
                self.df[self.price_col].pct_change()
            ) / (self.df["Volume"] * self.df[self.price_col] + 1e-10)

        # Price impact
        for window in [5, 20]:
            if "Volume" in self.df.columns:
                self.df[f"price_impact_{window}"] = self.df[
                    self.price_col
                ].pct_change().rolling(window).mean() / (
                    self.df["Volume"].rolling(window).mean() + 1e-10
                )

        return self.df

    def add_all_features(
        self,
        include_volume: bool = True,
        include_microstructure: bool = False,
        include_regime: bool = True,
    ) -> pd.DataFrame:
        """
        Add all available features to the dataframe.

        Args:
            include_volume (bool): Whether to include volume features
            include_microstructure (bool): Whether to include microstructure features
            include_regime (bool): Whether to include regime features

        Returns:
            pd.DataFrame: DataFrame with all features added
        """
        print("Adding returns...")
        self.add_returns()

        print("Adding moving averages...")
        self.add_moving_averages()

        print("Adding volatility features...")
        self.add_volatility_features()

        print("Adding momentum indicators...")
        self.add_momentum_indicators()

        print("Adding Bollinger Bands...")
        self.add_bollinger_bands()

        print("Adding statistical features...")
        self.add_statistical_features()

        print("Adding trend features...")
        self.add_trend_features()

        if include_volume and "Volume" in self.df.columns:
            print("Adding volume features...")
            self.add_volume_features()

        if include_microstructure:
            print("Adding microstructure features...")
            self.add_microstructure_features()

        if include_regime:
            print("Adding market regime features...")
            self.add_market_regime_features()

        print(f"Feature engineering complete. Total features: {len(self.df.columns)}")
        return self.df

    def get_features(self) -> pd.DataFrame:
        """
        Get the dataframe with engineered features.

        Returns:
            pd.DataFrame: DataFrame with all engineered features
        """
        return self.df

    def save_features(self, filepath: str):
        """
        Save engineered features to file.

        Args:
            filepath (str): Path to save the features (CSV or Parquet)
        """
        if filepath.endswith(".parquet"):
            self.df.to_parquet(filepath, index=True)
        else:
            self.df.to_csv(filepath, index=True)
        print(f"Features saved to {filepath}")


class MultiAssetFeatureEngineer:
    """
    Feature engineering for multiple assets (for master allocator environment).
    """

    def __init__(self, asset_dfs: Dict[str, pd.DataFrame]):
        """
        Initialize with multiple asset dataframes.

        Args:
            asset_dfs (Dict[str, pd.DataFrame]): Dictionary of {asset_name: dataframe}
        """
        self.asset_dfs = asset_dfs
        self.correlation_window = 60

    def add_cross_asset_features(self) -> pd.DataFrame:
        """
        Add cross-asset correlation and diversification features.

        Returns:
            pd.DataFrame: Combined dataframe with cross-asset features
        """
        # Calculate returns for all assets
        returns_dict = {}
        for asset_name, df in self.asset_dfs.items():
            returns_dict[asset_name] = df["Close"].pct_change()

        returns_df = pd.DataFrame(returns_dict)

        # Rolling correlation matrix
        rolling_corr = returns_df.rolling(self.correlation_window).corr()

        # Average correlation (diversification measure)
        avg_corr = rolling_corr.groupby(level=0).mean().mean(axis=1)

        # Create combined feature dataframe
        combined_df = pd.DataFrame(index=returns_df.index)
        combined_df["avg_correlation"] = avg_corr

        # Add individual asset returns
        for asset_name in returns_dict.keys():
            combined_df[f"{asset_name}_return"] = returns_dict[asset_name]
            combined_df[f"{asset_name}_volatility"] = returns_dict[asset_name].rolling(
                20
            ).std() * np.sqrt(252)

        return combined_df


def main():
    """
    Example usage of the FeatureEngineer class.
    """
    # Create sample data
    import sys
    import os

    # Add parent directory to path for imports
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    from src.data.data_loader import DataLoader

    print("Loading sample data...")
    loader = DataLoader(["AAPL"], "2022-01-01", "2023-12-31")
    df_raw = loader.load_raw_data()

    print(f"Raw data shape: {df_raw.shape}")
    print(f"Raw data columns: {df_raw.columns.tolist()}")

    # Handle multi-ticker DataFrame structure from yfinance
    # When single ticker, yfinance still creates MultiIndex columns
    if isinstance(df_raw.columns, pd.MultiIndex):
        # Flatten the MultiIndex for single ticker
        df = df_raw.copy()
        df.columns = [
            "_".join(col).strip() if isinstance(col, tuple) else col
            for col in df_raw.columns.values
        ]
        # Find the Close column (will be like 'Close_AAPL')
        close_cols = [col for col in df.columns if col.startswith("Close")]
        if close_cols:
            price_col = close_cols[0]
        else:
            price_col = "Close"
    else:
        df = df_raw
        price_col = "Close"

    print(f"\nProcessed data columns: {df.columns.tolist()}")
    print(f"Using price column: {price_col}")

    print("\nEngineering features...")
    engineer = FeatureEngineer(df, price_col=price_col)
    df_features = engineer.add_all_features(
        include_volume=True, include_microstructure=True, include_regime=True
    )

    print("\nFeature summary:")
    print(df_features.info())
    print("\nSample features:")
    print(df_features.head())

    # Save features
    engineer.save_features("data/processed/sample_features.csv")
    print("\nFeatures saved successfully!")


if __name__ == "__main__":
    main()
