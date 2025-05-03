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


subtasks = []

for ticker in PINNACLE_ASSETS:
    print(ticker)
    srs = pd.read_csv(f"data/prices.csv", )
    srs = srs[srs["ticker"]==ticker]
    srs["daily_return"] = srs["close"] / srs["close"].shift(1) - 1
    srs["next_day_return"] = srs["daily_return"].shift(-1)
    srs = srs.dropna()
    srs["date"] = pd.to_datetime(srs["date"])
    srs = srs[srs.date < dt.datetime(TEST_YEAR_START, 1, 1)]

    changepoint_data = read_changepoint_results_and_fill_na(
        f"data/cpd_21lbw/{ticker}.csv", 21
    )

    # TODO this is a mess
    # changepoint_data["date"] = changepoint_data.index
    changepoint_data = changepoint_data.merge(srs, on="date")
    changepoint_data = changepoint_data.set_index("date")
    changepoint_data.index.name = "date"

    # print(changepoint_data)

    tasks = assign_tasks(changepoint_data)

    # get rid of boundary tasks for now
    tasks = tasks[tasks["task"] >= 0]
    task_count = tasks.groupby("task")["t"].count()

    for task_num, task in tasks.groupby("task"):
        # print(task_num)
        splits = split_dataframe(task, MAX_SERIES)
        for i in range(len(splits)):
            splits[i] = splits[i].assign(subtask=i).assign(ticker=ticker)
            vol_scaling = TARGET_VOLATILITY / annual_volatility(
                splits[i]["daily_return"]
            )
            splits[i]["daily_return"] *= vol_scaling
            splits[i]["next_day_return"] *= vol_scaling
            # print(splits[i]["daily_return"].std()*np.sqrt(252))
        subtasks += splits

all_tasks = pd.concat(subtasks)
unique_tasks = (
    all_tasks[["task", "subtask", "ticker"]].drop_duplicates().reset_index(drop=True)
)
unique_tasks["set"] = unique_tasks.index
all_tasks["date"] = all_tasks.index
all_tasks = all_tasks.merge(unique_tasks, on=["task", "subtask", "ticker"])
# all_tasks


# train = all_tasks[all_tasks.date < dt.datetime(TEST_YEAR_START, 1, 1)]
train = all_tasks.groupby("set").filter(lambda x: len(x) >= MIN_SERIES)


# test = all_tasks.groupby("set").filter(
#     lambda row: row["date"].max() >= dt.datetime(TEST_YEAR_START, 1, 1)
# )


# def query_set_train(set_num: int):
#     data = train[train["set"]==set_num]
#     # print(data)
#     context_x = np.reshape(np.array([[data["daily_return"].tolist()]]), [1,len(data),1])
#     context_y = np.reshape(np.array([[data["next_day_return"].tolist()]]), [1,len(data),1])
#     target_x = np.array([[[float(len(data))]]])
#     target_y = np.array([[[data["next_day_return"].iloc[-1]]]])
#     query = (torch.tensor(context_x, dtype=torch.float32), torch.tensor(context_y, dtype=torch.float32)), torch.tensor(target_x, dtype=torch.float32)
#     return query, torch.tensor(target_y, dtype=torch.float32)
# query, target_y = query_set_train(0)

# train_data_prepped = pd.Series(train["set"].unique()).map(query_set_train)


def query_set_train_all_segments(set_num: int):
    segments = []
    data_whole = train[train["set"] == set_num]
    for length in range(MIN_SERIES, len(data_whole) + 1):
        # TODO redo volatility yet again int here
        data = data_whole.iloc[0:length].copy()
        vol_scaling = TARGET_VOLATILITY / annual_volatility(data["daily_return"])
        data["daily_return"] *= vol_scaling
        data["next_day_return"] *= vol_scaling
        context_x = np.reshape(
            np.array([[data["daily_return"].tolist()]]), [1, len(data), 1]
        )
        context_y = np.reshape(
            np.array([[data["next_day_return"].tolist()]]), [1, len(data), 1]
        )
        target_x = np.array([[[float(len(data))]]])
        target_y = np.array([[[data["next_day_return"].iloc[-1]]]])
        query = (
            torch.tensor(context_x, dtype=torch.float32),
            torch.tensor(context_y, dtype=torch.float32),
        ), torch.tensor(target_x, dtype=torch.float32)
        segments.append((query, torch.tensor(target_y, dtype=torch.float32), len(data)))
    return segments


def context_all_segments(set_num: int):
    segments = []
    data_whole = train[train["set"] == set_num]
    for length in range(MIN_SERIES, len(data_whole) + 1):
        # TODO redo volatility yet again int here
        data = data_whole.iloc[0:length].copy()
        date = data["date"].iloc[-1]
        vol_scaling = TARGET_VOLATILITY / annual_volatility(data["daily_return"])
        data["daily_return"] *= vol_scaling
        data["next_day_return"] *= vol_scaling
        context_x = np.reshape(
            np.array([[data["daily_return"].tolist()]]), [1, len(data), 1]
        )
        context_y = np.reshape(
            np.array([[data["next_day_return"].tolist()]]), [1, len(data), 1]
        )
        # target_x = np.array([[[float(len(data))]]])
        # target_y = np.array([[[data["next_day_return"].iloc[-1]]]])
        context = (
            torch.tensor(context_x, dtype=torch.float32),
            torch.tensor(context_y, dtype=torch.float32),
            set_num,
            len(data),
            date,
        )
        segments.append(context)
    return segments


segments_and_ticker = train[["set", "ticker"]].drop_duplicates().reset_index(drop=True)

train_data_prepped_all_segments = pd.DataFrame(
    pd.Series(
        pd.Series(segments_and_ticker["set"]).map(context_all_segments).sum()
    ).tolist(),
    columns=["x", "y", "set", "seq_len", "date"],
).merge(segments_and_ticker, on="set")


train_data_prepped_all_segments.to_pickle(f"data/prepped.pkl")

# dfs = []
# for i, group in train_data_prepped_all_segments.groupby(["ticker", "seq_len"]):
#     if len(group) % 2:
#         group = group.iloc[:-1]
#     context = group.iloc[::2, :].rename(columns={"x": "context_x", "y": "context_y"}).reset_index(drop=True)
#     target = group.iloc[1::2, :][["x", "y"]].rename(columns={"x": "target_x", "y": "target_y"}).reset_index(drop=True)
#     dfs.append(pd.concat([context, target], axis=1))

# queries = pd.concat(dfs).reset_index()

# queries.to_pickle(f"data/prepped.pkl")
    
# counts = (queries["seq_len"].value_counts() // BATCH_SIZE).tail(10)


# rows = (
#     queries[["seq_len"]]
#     .groupby("seq_len")
#     .sample(frac=1)
#     .groupby("seq_len")
#     .apply(lambda x: x.iloc[: -(x.count()[0] % BATCH_SIZE)])
#     .index.map(lambda i: i[1])
#     .tolist()
# )

# queries_for_batch = queries.loc[rows]


# train_data_batched = []
# for chunk in range(len(queries_for_batch) // BATCH_SIZE):
#     batch_data = queries_for_batch.iloc[
#         range(chunk * BATCH_SIZE, (chunk + 1) * BATCH_SIZE)
#     ]

#     context_x = torch.cat(batch_data["context_x"].tolist())
#     context_y = torch.cat(batch_data["context_y"].tolist())
#     target_x = torch.cat(batch_data["target_x"].tolist())
#     target_y = torch.cat(batch_data["target_y"].tolist())
#     query = ((context_x, context_y), target_x)
#     train_data_batched.append((query, target_y))


# train_data_batched_prepped = pd.Series(train_data_batched)
# train_data_batched_prepped.to_pickle(f"data/train_data_prepped_all_segments.pkl")


# prediction_days = test[test["date"] >= dt.datetime(TEST_YEAR_START, 1, 1)][
#     ["date", "set"]
# ].rename(columns={"date": "prediction_date"})
# prediction_days


# test_data = prediction_days.merge(test, on="set")
# test_data = test_data[test_data["prediction_date"] > test_data["date"]]


# def query_set_test(pred_date, set_num):
#     data = test_data[
#         (test_data["set"] == set_num) & (test_data["prediction_date"] == pred_date)
#     ]
#     # print(data)
#     context_x = np.reshape(
#         np.array([[list(range(len(data)))]], dtype=float), [1, len(data), 1]
#     )
#     context_y = np.reshape(
#         np.array([[data["daily_return"].tolist()]]), [1, len(data), 1]
#     )
#     target_x = np.array([[[float(len(data))]]])
#     target_y = np.array([[[data["next_day_return"].iloc[-1]]]])
#     query = (
#         torch.tensor(context_x, dtype=torch.float32),
#         torch.tensor(context_y, dtype=torch.float32),
#     ), torch.tensor(target_x, dtype=torch.float32)
#     return query, torch.tensor(target_y, dtype=torch.float32)


# test_data_prepped = (
#     test_data[["prediction_date", "set"]]
#     .drop_duplicates()
#     .apply(lambda row: query_set_test(row["prediction_date"], row["set"]), axis=1)
# )
# test_data_prepped


# from empyrical import sharpe_ratio


# EPOCHS = int(5)
# SIZE = 64
# # MAX_CONTEXT_POINTS = 10
# # PLOT_AFTER = int(2e4)
# LR = 1e-4

# # Sizes of the layers of the MLPs for the encoder and decoder
# # The final output layer of the decoder outputs two values, one for the mean and
# # one for the variance of the prediction at the target location
# # encoder_output_sizes = [SIZE, SIZE, SIZE, SIZE]
# # decoder_output_sizes = [SIZE, SIZE, 2]

# # # Define the model
# # encoder_input_size = 1 + 1  # x and y pairs are encoder into the context
# # decoder_input_size = (
# #     SIZE + 1
# # )  # target is concatenated onto the representation as input into the decoder


# model = RecurentAttentiveNeuralProcess(
#     1,
#     1,
#     64,
#     64,
#     use_lstm_de=True,
#     use_lstm_le=True,
#     use_lstm_d=True,
#     use_rnn=False,
#     use_self_attn=True,
#     det_enc_cross_attn_type="multihead",
#     det_enc_self_attn_type="multihead",
#     latent_enc_self_attn_type="multihead",
# ).cuda()

# # Set up the optimizer and train step
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# for it in range(EPOCHS):
#     print(it)

#     train_loss = []
#     for query, target_y in train_data_batched_prepped.sample(frac=1):
#         # Train dataset
#         # dataset_train = GPCurvesReader(
#         #     batch_size=64, max_num_context=MAX_CONTEXT_POINTS)
#         # data_train = dataset_train.generate_curves()

#         # # Test dataset
#         # dataset_test = GPCurvesReader(
#         #     batch_size=1, max_num_context=MAX_CONTEXT_POINTS, testing=True)
#         # data_test = dataset_test.generate_curves()

#         y_pred, loss_dict, dist_dict  = model(
#                 query[0][0].cuda(), query[0][1].cuda(), query[1].cuda(), target_y.cuda()
#             )
        
#         optimizer.zero_grad()
#         loss = loss_dict["loss"]
#         loss.backward()
#         train_loss.append(float(loss.detach().cpu().numpy()))
#         optimizer.step()
        
    
#     print(f"train loss: {np.mean(train_loss)}")

#     test_losses = []
#     realised_returns = []
#     returns = []
#     # for query, target_y in test_data_prepped.sample(frac=1):
#     #     log_prob, pred_y, var = model.forward(
#     #         query,
#     #         1,
#     #         query[0][0].shape[1],
#     #         target_y,
#     #     )

#     #     test_losses.append(-torch.mean(log_prob).detach().item())
#     #     actual_return = target_y.cpu().detach().numpy()[0, 0, 0]
#     #     returns.append(actual_return)
#     #     realised_returns.append(
#     #         np.sign(pred_y.cpu().detach().numpy()[0, 0, 0]) * actual_return
#     #     )

#     print(np.mean(test_losses))

#     print(
#         sharpe_ratio(np.array(realised_returns)),
#         " vs baseline ",
#         sharpe_ratio(np.array(returns)),
#     )

#     # # Plot the predictions in `PLOT_AFTER` intervals
#     # if it % PLOT_AFTER == 0:

#     #     (context_x, context_y), target_x = data_test.query
#     #     # Get the predicted mean and variance at the target points for the testing set
#     #     log_prob, pred_y, var = model.forward(
#     #         data_test.query, data_test.num_total_points,
#     #         data_test.num_context_points, data_test.target_y)
#     #     test_loss = -torch.mean(log_prob).detach().item()
#     #     print('Iteration: {}, loss: {}'.format(it, test_loss))

#     #     # Plot the prediction and the context
#     #     plot_functions(target_x,
#     #                    data_test.target_y,
#     #                    context_x,
#     #                    context_y,
#     #                    pred_y.detach().numpy(),
#     #                    var.detach().numpy())


# test_losses
