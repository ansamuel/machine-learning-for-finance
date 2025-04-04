"""Functions for calculating financial returns and volatility."""

import numpy as np
import pandas as pd
from typing import Optional

from utils.constants import VOL_LOOKBACK, VOL_TARGET


def calculate_returns(prices: pd.Series, day_offset: int = 1) -> pd.Series:
    """Calculate returns over a specified time period.
    
    For each element in the price series, computes the returns over 
    the past number of days specified by offset.
    """
    return prices / prices.shift(day_offset) - 1.0


def calculate_daily_volatility(daily_returns: pd.Series) -> pd.Series:
    """Calculate daily volatility using exponential weighted moving average."""
    return daily_returns.ewm(span=VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std().bfill()


def calculate_volatility_scaled_returns(
    daily_returns: pd.Series, 
    daily_volatility: Optional[pd.Series] = None
) -> pd.Series:
    """Calculate volatility scaled returns for annualized volatility target.
    
    Scales returns to target the VOL_TARGET annualized volatility (15%).
    """
    if daily_volatility is None or len(daily_volatility) == 0:
        daily_volatility = calculate_daily_volatility(daily_returns)
        
    annualized_volatility = daily_volatility * np.sqrt(252)  # annualized
    return daily_returns * VOL_TARGET / annualized_volatility.shift(1)