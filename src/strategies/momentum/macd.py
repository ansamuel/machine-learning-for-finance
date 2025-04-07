"""MACD momentum strategy implementation."""

import numpy as np
import pandas as pd
from typing import List, Tuple

from utils.constants import TRADING_DAYS_PER_QUARTER, TRADING_DAYS_PER_YEAR


class MACDStrategy:
    """MACD (Moving Average Convergence Divergence) based momentum strategy.
    
    Implements the combined MACD signal for multiple short/long combinations,
    as described in https://arxiv.org/pdf/1904.04912.pdf
    """
    
    def __init__(self, trend_combinations: List[Tuple[int, int]] = None):
        """Initialize the MACD strategy.
        
        Args:
            trend_combinations: List of short/long trend combinations.
                               Defaults to [(8, 24), (16, 48), (32, 96)].
        """
        if trend_combinations is None:
            self.trend_combinations = [(8, 24), (16, 48), (32, 96)]
        else:
            self.trend_combinations = trend_combinations

    @staticmethod
    def calc_signal(prices: pd.Series, short_timescale: int, long_timescale: int) -> pd.Series:
        """Calculate MACD signal for a specific short/long timescale combination.
        
        Args:
            prices: Series of asset prices.
            short_timescale: Short timescale for exponential moving average.
            long_timescale: Long timescale for exponential moving average.
            
        Returns:
            MACD signal normalized by volatility.
        """
        def _calc_halflife(timescale):
            return np.log(0.5) / np.log(1 - 1 / timescale)

        macd = (
            prices.ewm(halflife=_calc_halflife(short_timescale)).mean()
            - prices.ewm(halflife=_calc_halflife(long_timescale)).mean()
        )
        q = macd / prices.rolling(TRADING_DAYS_PER_QUARTER).std().bfill()
        return q / q.rolling(TRADING_DAYS_PER_YEAR).std().bfill()

    @staticmethod
    def scale_signal(signal: pd.Series) -> pd.Series:
        """Scale the signal using a non-linear transformation.
        
        Applies a scaling function that reduces extreme signals,
        implementing the strategy's risk management approach.
        
        Args:
            signal: Raw MACD signal.
            
        Returns:
            Scaled signal.
        """
        return signal * np.exp(-(signal ** 2) / 4) / 0.89

    def calculate_combined_signal(self, prices: pd.Series) -> pd.Series:
        """Calculate combined MACD signal across all trend combinations.
        
        Args:
            prices: Series of asset prices.
            
        Returns:
            Combined MACD signal (average of all combinations).
        """
        return np.sum(
            [self.calc_signal(prices, short, long) 
             for short, long in self.trend_combinations]
        ) / len(self.trend_combinations)