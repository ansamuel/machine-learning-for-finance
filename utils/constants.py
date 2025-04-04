"""Shared financial constants for volatility calculations and data processing."""

# Volatility constants
VOL_LOOKBACK = 60  # Lookback window for ex-ante volatility calculation
VOL_TARGET = 0.15  # 15% annualized volatility target

# Winsorization constants
WINSORIZE_VOL_THRESHOLD = 5  # Multiple of standard deviations to winsorize by
WINSORIZE_HALFLIFE = 252  # Half-life for exponential weighting in winsorization

# Trading day constants
TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_QUARTER = 63
TRADING_DAYS_PER_HALF_YEAR = 126