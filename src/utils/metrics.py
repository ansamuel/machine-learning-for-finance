"""Functions for calculating performance metrics for trading strategies."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from empyrical import (
    sharpe_ratio,
    calmar_ratio,
    sortino_ratio,
    max_drawdown,
    downside_risk,
    annual_return,
    annual_volatility,
)
from utils.constants import VOL_TARGET


def calculate_performance_metrics(
    returns: pd.DataFrame, 
    metric_suffix: str = "", 
    num_identifiers: Optional[int] = None
) -> Dict[str, float]:
    """Calculate comprehensive performance metrics for strategy evaluation."""
    if not num_identifiers:
        num_identifiers = len(returns.dropna()["identifier"].unique())
    
    # Calculate mean returns across identifiers
    clean_returns = returns.dropna()
    grouped_returns = clean_returns.groupby(level=0)["captured_returns"]
    summed_returns = grouped_returns.sum()
    return_values = summed_returns / num_identifiers
    
    # Positive returns for ratio calculations
    positive_returns = return_values[return_values > 0.0]
    negative_returns = return_values[return_values < 0.0]
    
    # Calculate metrics
    metrics = {}
    metrics[f"annual_return{metric_suffix}"] = annual_return(return_values)
    metrics[f"annual_volatility{metric_suffix}"] = annual_volatility(return_values)
    metrics[f"sharpe_ratio{metric_suffix}"] = sharpe_ratio(return_values)
    metrics[f"downside_risk{metric_suffix}"] = downside_risk(return_values)
    metrics[f"sortino_ratio{metric_suffix}"] = sortino_ratio(return_values)
    metrics[f"max_drawdown{metric_suffix}"] = -max_drawdown(return_values)
    metrics[f"calmar_ratio{metric_suffix}"] = calmar_ratio(return_values)
    metrics[f"perc_pos_return{metric_suffix}"] = len(positive_returns) / len(return_values)
    
    # Calculate profit-loss ratio (avoid division by zero)
    if len(negative_returns) > 0 and np.mean(np.abs(negative_returns)) != 0:
        metrics[f"profit_loss_ratio{metric_suffix}"] = np.mean(positive_returns) / np.mean(np.abs(negative_returns))
    else:
        metrics[f"profit_loss_ratio{metric_suffix}"] = np.nan
    
    return metrics


def calculate_performance_metrics_subset(
    return_values: pd.Series, 
    metric_suffix: str = ""
) -> Dict[str, float]:
    """Calculate a subset of performance metrics for strategy evaluation."""
    metrics = {}
    metrics[f"annual_return{metric_suffix}"] = annual_return(return_values)
    metrics[f"annual_volatility{metric_suffix}"] = annual_volatility(return_values)
    metrics[f"downside_risk{metric_suffix}"] = downside_risk(return_values)
    metrics[f"max_drawdown{metric_suffix}"] = -max_drawdown(return_values)
    
    return metrics


def calculate_sharpe_by_year(
    returns: pd.DataFrame, 
    suffix: Optional[str] = None
) -> Dict[str, float]:
    """Calculate Sharpe ratio for each year in the dataset."""
    if suffix is None:
        suffix = ""

    # Add year column to the DataFrame
    returns_by_year = returns.copy()
    returns_by_year["year"] = returns_by_year.index.year
    
    # Extract relevant columns and remove NA values
    clean_returns = returns_by_year.dropna()[["year", "captured_returns"]]
    
    # Calculate mean returns by date
    mean_by_date = clean_returns.groupby(level=0).mean()
    
    # Group by year
    year_groups = mean_by_date.groupby("year")
    
    # Calculate Sharpe ratio for each year
    sharpes = {}
    for year, year_data in year_groups:
        year_sharpe = sharpe_ratio(year_data["captured_returns"])
        key = f"sharpe_ratio_{int(year)}{suffix}"
        sharpes[key] = year_sharpe
    
    return sharpes


def calculate_net_returns(
    positions: pd.DataFrame, 
    list_basis_points: List[float], 
    identifiers: Optional[List[str]] = None
) -> pd.DataFrame:
    """Calculate net returns after transaction costs."""
    if not identifiers:
        identifiers = positions["identifier"].unique().tolist()
    
    # Convert basis points to decimal
    cost = np.atleast_2d(list_basis_points) * 1e-4
    result_frames = []
    
    # Process each identifier separately
    for identifier in identifiers:
        # Filter data for this identifier
        position_slice = positions[positions["identifier"] == identifier].reset_index(drop=True)
        
        # Calculate annualized volatility and scaled positions
        annualized_vol = position_slice["daily_vol"] * np.sqrt(252)
        scaled_position = VOL_TARGET * position_slice["position"] / annualized_vol
        
        # Calculate transaction costs (position changes)
        position_changes = scaled_position.diff().abs().fillna(0.0)
        transaction_costs = position_changes.to_frame().to_numpy() * cost
        
        # Calculate net returns after costs
        captured_returns = position_slice[["captured_returns"]].to_numpy()
        net_captured_returns = captured_returns - transaction_costs
        
        # Create column names for different basis points
        columns = []
        for bp in list_basis_points:
            bp_str = str(bp).replace('.', '_')
            columns.append(f"captured_returns_{bp_str}_bps")
        
        # Create DataFrame with net returns
        net_returns = pd.DataFrame(net_captured_returns, columns=columns)
        
        # Combine original data with net returns
        combined = pd.concat([position_slice, net_returns], axis=1)
        result_frames.append(combined)
    
    # Combine results for all identifiers
    return pd.concat(result_frames).reset_index(drop=True)