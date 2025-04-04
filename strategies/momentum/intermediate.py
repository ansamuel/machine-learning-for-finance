"""Intermediate momentum strategy implementation."""

import numpy as np
import pandas as pd
from typing import Optional

from utils.constants import TRADING_DAYS_PER_MONTH, TRADING_DAYS_PER_YEAR
from utils.returns import (
    calculate_returns,
    calculate_volatility_scaled_returns,
)


class IntermediateStrategy:
    """Implements the intermediate momentum strategy.
    
    This strategy combines monthly and annual signals with a weighted average.
    When w=0, this is equivalent to the Moskowitz TSMOM strategy.
    
    See https://arxiv.org/pdf/2105.13727.pdf
    """
    
    def __init__(self, weight: float = 0.5, volatility_scaling: bool = True):
        """Initialize the intermediate strategy.
        
        Args:
            weight: Weight for monthly signal, where 1-weight applies to annual signal.
                   When weight=0, this is equivalent to Moskowitz TSMOM.
            volatility_scaling: Whether to scale returns by volatility.
        """
        self.weight = weight
        self.volatility_scaling = volatility_scaling
    
    def calculate_signal(self, prices: pd.Series) -> pd.Series:
        """Calculate strategy signal based on price data.
        
        Args:
            prices: Series of asset prices.
            
        Returns:
            Series of strategy signals for each time period.
        """
        daily_returns = calculate_returns(prices)
        monthly_returns = calculate_returns(prices, TRADING_DAYS_PER_MONTH)
        annual_returns = calculate_returns(prices, TRADING_DAYS_PER_YEAR)
        
        next_day_returns = (
            calculate_volatility_scaled_returns(daily_returns).shift(-1)
            if self.volatility_scaling
            else daily_returns.shift(-1)
        )
        
        return (
            self.weight * np.sign(monthly_returns) * next_day_returns
            + (1 - self.weight) * np.sign(annual_returns) * next_day_returns
        )