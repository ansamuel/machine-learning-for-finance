import pandas as pd
import numpy as np
from empyrical import annual_volatility

import os

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
import matplotlib.pyplot as plt
import collections
import datetime as dt


MAX_SERIES = 63
MIN_SERIES = 3
BATCH_SIZE = 32

TARGET_VOLATILITY = np.sqrt(252)


TEST_YEAR_START = 2015

TEST_END = 2020

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



def assign_tasks(
    changepoint_data: pd.DataFrame,
    changepoint_threshold: float = 0.995,
    burn_in: int = 5,
) -> pd.DataFrame:

    boundaries = changepoint_data[
        (changepoint_data["cp_score"] >= changepoint_threshold)
    ]

    last_location = boundaries.iloc[0]["cp_location"]
    cp_locations = [last_location]

    data_w_tasks = changepoint_data.copy()
    data_w_tasks["task"] = np.NaN

    data_w_tasks.loc[boundaries.index[0], "task"] = 0
    task_number = 1

    for idx, row in boundaries.iloc[1:, :].iterrows():
        # print(idx)
        if row["cp_location"] - last_location >= burn_in:
            data_w_tasks.loc[idx, "task"] = task_number
            # last_change_date = row["date"]
            last_location = row["cp_location"]
            cp_locations.append(last_location)
            task_number += 1

    data_w_tasks = data_w_tasks.bfill().fillna(task_number)
    data_w_tasks["task"] = data_w_tasks["task"].astype(int)

    def boundary_task(l):
        last_before = int(np.floor(l))
        return (
            changepoint_data.reset_index()
            .set_index("t")
            .loc[list(range(last_before - burn_in + 1, last_before + burn_in + 1))]
        )

    cp_loc_srs = pd.Series(cp_locations)
    cp_loc_srs = cp_loc_srs[cp_loc_srs >= changepoint_data["t"].min() + burn_in]
    # print(cp_loc_srs)

    boundary_tasks = cp_loc_srs.map(boundary_task)
    # start_task = data_w_tasks["task"].max() + 1
    # list(range(start_task, start_task + len(boundary_tasks)))

    for i in range(len(boundary_tasks)):
        boundary_tasks.iloc[i] = (
            boundary_tasks.iloc[i]
            .assign(task=(-i - 1))
            .reset_index()
            .set_index("date")[
                ["t", "cp_location", "cp_location_norm", "cp_score", "task"]
            ]
        )

    return pd.concat([data_w_tasks] + boundary_tasks.tolist())


def split_dataframe(df, max_series=MAX_SERIES, min_series=MIN_SERIES):
    chunks = list()
    if len(df) % max_series < min_series:
        num_chunks = len(df) // max_series
    else:
        num_chunks = len(df) // max_series + 1
    for i in range(num_chunks):
        if i == num_chunks - 1:
            chunks.append(df.iloc[i * max_series :])
        else:
            chunks.append(df.iloc[i * max_series : (i + 1) * max_series])
    return chunks

def read_changepoint_results_and_fill_na(
    file_path: str, lookback_window_length: int
) -> pd.DataFrame:
    """Read output data from changepoint detection module into a dataframe.
    For rows where the module failed, information for changepoint location and severity is
    filled using the previous row.


    Args:
        file_path (str): the file path of the csv containing the results
        lookback_window_length (int): lookback window length - necessary for filling in the blanks for norm location

    Returns:
        pd.DataFrame: changepoint severity and location information
    """

    return (
        pd.read_csv(file_path, index_col=0, parse_dates=True)
        .fillna(method="ffill")
        .dropna()  # if first values are na
        .assign(
            cp_location_norm=lambda row: (row["t"] - row["cp_location"])
            / lookback_window_length
        )  # fill by assigning the previous cp and score, then recalculate norm location
    )


def assign_tasks(
    changepoint_data: pd.DataFrame, changepoint_threshold: float = 0.995, burn_in: int = 5
) -> pd.DataFrame:

    boundaries = changepoint_data[
        (changepoint_data["cp_score"] >= changepoint_threshold)
    ]

    last_location = boundaries.iloc[0]["cp_location"]
    data_w_tasks = changepoint_data.copy()
    data_w_tasks["task"] = np.NaN

    data_w_tasks.loc[boundaries.index[0], "task"] = 0
    task_number = 1

    for idx, row in boundaries.iloc[1:, :].iterrows():
        # print(idx)
        if row["cp_location"] - last_location >= burn_in:
            data_w_tasks.loc[idx, "task"] = task_number
            # last_change_date = row["date"]
            last_location = row["cp_location"]
            task_number += 1

    data_w_tasks = data_w_tasks.bfill().fillna(task_number)
    data_w_tasks["task"] = data_w_tasks["task"].astype(int)
    return data_w_tasks


sets = []
seq_lens = []
x_vals = []
y_vals = []
dates = []
tickers = []
count = 0

for ticker in PINNACLE_ASSETS:
    print(ticker)
    srs = pd.read_csv(f"data/prices.csv", )
    srs = srs[srs["ticker"]==ticker]
    srs["daily_return"] = srs["close"] / srs["close"].shift(1) - 1
    srs["next_day_return"] = srs["daily_return"].shift(-1)
    srs = srs.dropna()
    srs["date"] = pd.to_datetime(srs["date"])

    #TODO do buffer properly based on end date
    # srs = srs[srs.date >= dt.datetime(TEST_YEAR_START - 1, 1, 1)]

    changepoint_data = read_changepoint_results_and_fill_na(
        f"data/cpd_21lbw/{ticker}.csv", 21
    )

    # TODO this is a mess
    # changepoint_data["date"] = changepoint_data.index
    changepoint_data = changepoint_data.merge(srs, on="date")
    # changepoint_data = changepoint_data.set_index("date")
    # changepoint_data.index.name = "date"

    # print(changepoint_data)

    tasks = assign_tasks(changepoint_data)

    # get rid of boundary tasks for now
    tasks = tasks[tasks["task"] >= 0]
    task_count = tasks.groupby("task")["t"].count()

    prediction_days = tasks[tasks["date"] >= dt.datetime(TEST_YEAR_START, 1, 1)][
        ["date", "task"]
    ].rename(columns={"date": "prediction_date"})

    tasks = prediction_days.merge(tasks, on="task")
    tasks = tasks[tasks["prediction_date"] > tasks["date"]]
    tasks = tasks[tasks.groupby(["prediction_date", "task"])['date'].transform('size').ge(MIN_SERIES)]


    for key, data in tasks.groupby(["prediction_date", "task"]):
        data = data[-MAX_SERIES:].copy()
        vol_scaling = TARGET_VOLATILITY / annual_volatility(
            data["daily_return"]
        )
        data["daily_return"] *= vol_scaling
        data["next_day_return"] *= vol_scaling
            # print(splits[i]["daily_return"].std()*np.sqrt(252))

        seq_len = len(data)

        seq_lens.append(seq_len)
        
        x_vals.append(np.reshape(
            np.array([[data["daily_return"].tolist()]]), [1, len(data), 1]
        ))
        y_vals.append(np.reshape(
            np.array([[data["next_day_return"].tolist()]]), [1, len(data), 1]
        ))
        sets.append(count)
        tickers.append(ticker)
        dates.append(key[0])
        count +=1


test_data_prepped_all_segments = pd.DataFrame(
    {
        'x': x_vals,
        'y': y_vals,
        'set': sets,
        'seq_len': seq_lens,
        'date': dates,
        'ticker': tickers
    }
)

test_data_prepped_all_segments["x"] = test_data_prepped_all_segments["x"].map(lambda x: torch.tensor(x, dtype=torch.float32))
test_data_prepped_all_segments["y"] = test_data_prepped_all_segments["y"].map(lambda y: torch.tensor(y, dtype=torch.float32))

test_data_prepped_all_segments.to_pickle(f"data/prepped_test.pkl")
