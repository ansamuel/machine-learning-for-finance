"""Functions for creating and combining features."""

import numpy as np
import pandas as pd

from strategies.momentum.macd import MACDStrategy
from utils.returns import (
    calculate_returns as calc_returns,
    calculate_daily_volatility as calc_daily_vol,
    calculate_volatility_scaled_returns as calc_vol_scaled_returns,
)
from features.input import load_changepoint_data

# Constants
WINSORIZE_VOL_THRESHOLD = 5  # multiple to winsorise by
WINSORIZE_HALFLIFE = 252


def create_momentum_features(price_data: pd.DataFrame) -> pd.DataFrame:
    """Create momentum strategy features from price data."""
    # Filter out invalid prices
    features = price_data[
        ~price_data["close"].isna()
        | ~price_data["close"].isnull()
        | (price_data["close"] > 1e-8)  # price is zero
    ].copy()

    # Winsorize using rolling 5X standard deviations to remove outliers
    features["srs"] = features["close"]
    ewm = features["srs"].ewm(halflife=WINSORIZE_HALFLIFE)
    means = ewm.mean()
    stds = ewm.std()
    features["srs"] = np.minimum(features["srs"], means + WINSORIZE_VOL_THRESHOLD * stds)
    features["srs"] = np.maximum(features["srs"], means - WINSORIZE_VOL_THRESHOLD * stds)

    # Calculate returns and volatility
    features["daily_returns"] = calc_returns(features["srs"])
    features["daily_vol"] = calc_daily_vol(features["daily_returns"])
    
    # Vol scaling and shift to be next day returns
    features["target_returns"] = calc_vol_scaled_returns(
        features["daily_returns"], features["daily_vol"]
    ).shift(-1)

    # Add normalized returns features at different time scales
    # Daily normalized returns
    features["norm_daily_return"] = (
        calc_returns(features["srs"], 1)
        / features["daily_vol"] 
        / np.sqrt(1)
    )
    
    # Monthly normalized returns
    features["norm_monthly_return"] = (
        calc_returns(features["srs"], 21)
        / features["daily_vol"]
        / np.sqrt(21)
    )
    
    # Quarterly normalized returns
    features["norm_quarterly_return"] = (
        calc_returns(features["srs"], 63)
        / features["daily_vol"]
        / np.sqrt(63)
    )
    
    # Biannual normalized returns
    features["norm_biannual_return"] = (
        calc_returns(features["srs"], 126)
        / features["daily_vol"]
        / np.sqrt(126)
    )
    
    # Annual normalized returns
    features["norm_annual_return"] = (
        calc_returns(features["srs"], 252)
        / features["daily_vol"]
        / np.sqrt(252)
    )

    # Calculate MACD signals
    trend_combinations = [(8, 24), (16, 48), (32, 96)]
    for short_window, long_window in trend_combinations:
        features[f"macd_{short_window}_{long_window}"] = MACDStrategy.calc_signal(
            features["srs"], short_window, long_window
        )

    # Add date features
    if len(features):
        features["day_of_week"] = features.index.dayofweek
        features["day_of_month"] = features.index.day
        features["week_of_year"] = features.index.isocalendar().week
        features["month_of_year"] = features.index.month
        features["year"] = features.index.year
        features["date"] = features.index
    else:
        features["day_of_week"] = []
        features["day_of_month"] = []
        features["week_of_year"] = []
        features["month_of_year"] = []
        features["year"] = []
        features["date"] = []
        
    return features.dropna()


def merge_with_changepoint_features(
    momentum_features: pd.DataFrame, 
    cpd_folder_path: str, 
    lookback_window_length: int
) -> pd.DataFrame:
    """Merge momentum features with changepoint detection features."""
    # Load changepoint data
    changepoint_data = load_changepoint_data(cpd_folder_path, lookback_window_length)
    
    # Prepare the changepoint data for merging
    cp_features = changepoint_data[["ticker", "cp_location_norm", "cp_score"]].reset_index()
    
    # Rename columns to include the lookback window length
    cp_features = cp_features.rename(
        columns={
            "cp_location_norm": f"cp_rl_{lookback_window_length}",
            "cp_score": f"cp_score_{lookback_window_length}",
        }
    )
    
    # Merge the features
    merged_features = momentum_features.merge(
        cp_features,
        on=["date", "ticker"],
    )
    
    # Set the index back to date
    merged_features.index = merged_features["date"]
    
    return merged_features