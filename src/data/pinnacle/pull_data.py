"""Functions for loading Pinnacle financial data."""

import os
from typing import List

import pandas as pd
import numpy as np

from data.pinnacle.constants import PINNACLE_DATA_CUT, PINNACLE_DATA_FOLDER


def load_ticker_prices(ticker: str) -> pd.DataFrame:
    """Load price data for a single ticker.
    
    Args:
        ticker: Asset ticker symbol
        
    Returns:
        DataFrame with close prices indexed by date
    """
    file_path = os.path.join(PINNACLE_DATA_FOLDER, f"{ticker}_{PINNACLE_DATA_CUT}.CSV")
    
    # Read CSV with column names and parse dates
    prices = pd.read_csv(
        file_path,
        names=["date", "open", "high", "low", "close", "volume", "open_int"],
        parse_dates=[0],
        index_col=0,
    )
    
    # Extract close prices and replace zeros with NaN
    close_prices = prices[["close"]]
    close_prices = close_prices.replace(0.0, np.nan)
    
    return close_prices


def _fill_missing_values(prices: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in a price series.
    
    Fills forward values between first and last valid index.
    """
    first_valid = prices["close"].first_valid_index()
    last_valid = prices["close"].last_valid_index()
    
    # Extract the valid range and forward-fill
    valid_range = prices[first_valid:last_valid]
    return valid_range.ffill()


def load_multiple_ticker_prices(tickers: List[str], fill_missing_dates: bool = False) -> pd.DataFrame:
    """Load price data for multiple tickers.
    
    Args:
        tickers: List of asset ticker symbols
        fill_missing_dates: Whether to fill in missing dates with forward-filled values
        
    Returns:
        DataFrame with close prices for all tickers
    """
    # Load prices for each ticker and concatenate
    all_ticker_prices = []
    for ticker in tickers:
        ticker_prices = load_ticker_prices(ticker)
        ticker_prices_with_id = ticker_prices.assign(ticker=ticker).copy()
        all_ticker_prices.append(ticker_prices_with_id)
    
    combined_prices = pd.concat(all_ticker_prices)

    if not fill_missing_dates:
        return combined_prices.dropna().copy()

    # Get unique dates from the data
    unique_dates = combined_prices.reset_index()[["date"]].drop_duplicates().sort_values("date")
    ticker_indexed_prices = combined_prices.reset_index().set_index("ticker")

    # Fill in missing dates for each ticker
    filled_prices_by_ticker = []
    for ticker in tickers:
        ticker_date_prices = unique_dates.merge(ticker_indexed_prices.loc[ticker], on="date", how="left")
        ticker_date_prices = ticker_date_prices.assign(ticker=ticker)
        filled_ticker_prices = _fill_missing_values(ticker_date_prices)
        filled_prices_by_ticker.append(filled_ticker_prices)
    
    # Combine and format the result
    result_prices = pd.concat(filled_prices_by_ticker)
    result_prices = result_prices.reset_index()
    result_prices = result_prices.set_index("date")
    result_prices = result_prices.drop(columns="index")
    
    return result_prices.copy()