import os
from typing import List

import pandas as pd

import numpy as np

PINNACLE_DATA_FOLDER = "dataset/CLCDATA/"
PINNACLE_DATA_CUT = "RAD"

def pull_pinnacle_data(ticker: str) -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(PINNACLE_DATA_FOLDER, f"{ticker}_{PINNACLE_DATA_CUT}.CSV"),
        names=["date", "open", "high", "low", "close", "volume", "open_int"],
        parse_dates=[0],
        index_col=0,
    )[["close"]].replace(0.0, np.nan)


def _fill_blanks(data: pd.DataFrame):
    return data[
        data["close"].first_valid_index() : data["close"].last_valid_index()
    ].fillna(
        method="ffill"
    )  # .interpolate()


def pull_pinnacle_data_multiple(
    tickers: List[str], fill_missing_dates=False
) -> pd.DataFrame:
    data = pd.concat(
        [pull_pinnacle_data(ticker).assign(ticker=ticker).copy() for ticker in tickers]
    )

    if not fill_missing_dates:
        return data.dropna().copy()

    dates = data.reset_index()[["date"]].drop_duplicates().sort_values("date")
    data = data.reset_index().set_index("ticker")

    return (
        pd.concat(
            [
                _fill_blanks(
                    dates.merge(data.loc[t], on="date", how="left").assign(ticker=t)
                )
                for t in tickers
            ]
        )
        .reset_index()
        .set_index("date")
        .drop(columns="index")
        .copy()
    )

PINNACLE_ASSETS = [
    "AN",
    "BN",
    "CA",
    "CC",
    "CN",
    "DA",
    "DT",
    "DX",
    "EN",
    "ER",
    "ES",
    "FB",
    "FN",
    "GI",
    "JN",
    "JO",
    "KC",
    "KW",
    "LB",
    "LX",
    "MD",
    "MP",
    "NK",
    "NR",
    "SB",
    "SC",
    "SN",
    "SP",
    "TY",
    "UB",
    "US",
    "XU",
    "XX",
    "YM",
    "ZA",
    "ZC",
    "ZF",
    "ZG",
    "ZH",
    "ZI",
    "ZK",
    "ZL",
    "ZN",
    "ZO",
    "ZP",
    "ZR",
    "ZT",
    "ZU",
    "ZW",
    "ZZ",
]
pull_pinnacle_data_multiple(PINNACLE_ASSETS).to_csv("prices.csv")
