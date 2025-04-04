"""Constants and configuration for Pinnacle data source.

Contains folder paths, data filenames, asset lists and mappings.
"""

import os

# Data source configuration
PINNACLE_DATA_CUT = "RAD"
PINNACLE_DATA_FOLDER = os.path.join("dataset", "pinnacle", "CLCDATA")
PINNACLE_CPD_FOLDER = os.path.join("dataset", "pinnacle", "CPD")

# List of assets to process
PINNACLE_ASSETS = [
    "AN", "BN", "CA", "CC", "CN", "DA", "DT", "DX", "EN", "ER", "ES", "FB", "FN", "GI",
    "JN", "JO", "KC", "KW", "LB", "LX", "MD", "MP", "NK", "NR", "SB", "SC", "SN", "SP",
    "TY", "UB", "US", "XU", "XX", "YM", "ZA", "ZC", "ZF", "ZG", "ZH", "ZI", "ZK", "ZL",
    "ZN", "ZO", "ZP", "ZR", "ZT", "ZU", "ZW", "ZZ"
]

# Asset class mappings (FI: Fixed Income, CM: Commodities, FX: Forex, EQ: Equities)
PINNACLE_ASSET_CLASS_MAPPING = {
    "TY": "FI", "US": "FI", "FB": "FI", "UB": "FI", "DT": "FI", "ZA": "CM", "ZC": "CM",
    "ZG": "CM", "ZL": "CM", "ZW": "CM", "ZI": "CM", "ZP": "CM", "ZR": "CM", "ZZ": "CM",
    "KW": "CM", "ZT": "CM", "ZF": "CM", "ZK": "CM", "GI": "CM", "ZO": "CM", "ZH": "CM",
    "NR": "CM", "ZN": "CM", "ZU": "CM", "LB": "CM", "JO": "CM", "KC": "CM", "CC": "CM",
    "SB": "CM", "DA": "CM", "NK": "FX", "DX": "FX", "AN": "FX", "BN": "FX", "SN": "FX",
    "JN": "FX", "FN": "FX", "CN": "FX", "MP": "FX", "ER": "EQ", "XX": "EQ", "YM": "EQ",
    "ES": "EQ", "EN": "EQ", "SC": "EQ", "SP": "EQ", "MD": "EQ", "CA": "EQ", "XU": "EQ",
    "LX": "EQ", "AD": "FX", "AP": "FI", "AX": "EQ", "BC": "CM", "BG": "CM", "BO": "CM",
    "CB": "FX", "CL": "CM", "CT": "CM", "C_": "CM", "FA": "FI", "FC": "CM", "FX": "FX",
    "GC": "CM", "GS": "FI", "HG": "CM", "HO": "CM", "HS": "EQ", "LC": "CM", "LH": "CM",
    "MW": "CM", "NG": "CM", "O_": "CM", "PA": "CM", "PL": "CM", "RB": "CM", "SF": "FX",
    "SI": "CM", "SM": "CM", "S_": "CM", "TA": "FI", "TD": "FI", "TU": "FI", "UA": "FI",
    "W_": "CM", "ZB": "CM", "ZM": "CM", "ZS": "CM"
}