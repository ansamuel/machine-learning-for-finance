"""Functions for loading input data."""

import os
import pandas as pd


def load_changepoint_data(folder_path: str, lookback_window_length: int) -> pd.DataFrame:
    """Load changepoint detection data for all assets and normalize the location values."""
    changepoint_dfs = []
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        ticker = os.path.splitext(filename)[0]
        
        # Read and process the changepoint file
        changepoint = pd.read_csv(file_path, index_col=0, parse_dates=True)
        changepoint = changepoint.fillna(method="ffill")
        changepoint = changepoint.dropna()  # If first values are NA
        
        # Normalize changepoint location
        changepoint["cp_location_norm"] = (
            changepoint["t"] - changepoint["cp_location"]
        ) / lookback_window_length
        
        # Add ticker identifier
        changepoint["ticker"] = ticker
        
        changepoint_dfs.append(changepoint)
    
    return pd.concat(changepoint_dfs)