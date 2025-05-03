from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
import datetime as dt
import torch
from rnn_np import ANP_RNN_Model
import os
import json


BATCH_SIZE = 16
NUM_CONTEXT = 3
ITERATIONS = 100
MIN_SEQ_LEN = 5

N = 50

LR = 0.001

# LEN_CONTEXT = 6
# BATCH_SIZE = 256
ENCODER_OUTPUT_SIZES = [64, 64]
DECODER_OUTPUT_SIZES = [64, 64]
EMBEDDING_DIM = 64
LATENT_DIM = 4
# LATENT_DIM = 4
USE_X_ATTENTION = True
USE_SELF_ATTENTION = True
RUN_NAME = "test_pos_lre-3-4lat-5min"

TEST_START_YEAR = 2018  # valid 2015-208


print(f"Start training {RUN_NAME}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data_prepped_all_segments = pd.read_pickle("data/prepped.pkl")
train_data_prepped_all_segments = train_data_prepped_all_segments[
    train_data_prepped_all_segments["seq_len"] >= MIN_SEQ_LEN
].copy()

test_data_prepped_all_segments = pd.read_pickle("data/prepped_test.pkl")
test_data_prepped_all_segments = test_data_prepped_all_segments[
    test_data_prepped_all_segments["seq_len"] >= MIN_SEQ_LEN
].copy()
test_data_prepped_all_segments = test_data_prepped_all_segments.set_index(
    ["seq_len", "ticker", "date"]
)


num_target_sets = (
    train_data_prepped_all_segments.groupby(["seq_len", "ticker"])[["date"]]
    .count()
    .rename(columns={"date": "count"})
) // (NUM_CONTEXT + 1)

num_target_sets = num_target_sets[num_target_sets["count"] > 0].copy()

num_context_sets = num_target_sets * NUM_CONTEXT

contexts = []
targets = []

train_data_prepped_all_segments = train_data_prepped_all_segments.set_index(
    ["seq_len", "ticker"]
)

group_var = 0
for key in num_target_sets.index:
    count = num_target_sets.loc[key][0]

    seq_len, ticker = key
    segments = train_data_prepped_all_segments.loc[key]
    # targets.append(segments.iloc[-count:].assign(group_var=group_var))
    # contexts.append(segments.iloc[:-count].assign(group_var=group_var))
    targets.append(segments.iloc[-count:].copy())
    con = segments.iloc[:-count].copy()
    con["x"] = con["x"].map(lambda x: x.unsqueeze(2))
    con["y"] = con["y"].map(lambda x: x.unsqueeze(2))
    contexts.append(con)
    # group_var += 1
    # dates should already be sorted

# because all uysed as context when testing
train_data_prepped_all_segments["x"] = train_data_prepped_all_segments["x"].map(
    lambda x: x.unsqueeze(2)
)
train_data_prepped_all_segments["y"] = train_data_prepped_all_segments["y"].map(
    lambda x: x.unsqueeze(2)
)

train_data_prepped_all_segments = train_data_prepped_all_segments.sort_index()


model = ANP_RNN_Model(
    x_dim=1,
    y_dim=1,
    encoder_rnn_hidden_size_list=ENCODER_OUTPUT_SIZES,
    decoder_rnn_hidden_size_list=DECODER_OUTPUT_SIZES,
    embedding_dim=EMBEDDING_DIM,
    latent_dim=LATENT_DIM,
    context_concat="stack",
    use_cross_attention=USE_X_ATTENTION,
    use_self_attention=USE_SELF_ATTENTION,
    de_cross_attention_type="dot",
    de_self_attention_type="dot",
    latent_path=True,
)

model = model.to(device)

# Set up the optimizer and train step
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# probably need to separate the target data out earlier

best_valid_sharpe = 0
for it in range(ITERATIONS):
    print(f"Iteration {it}")

    # train_mse = []
    captured_return = []

    # TODO this is a mess
    dfs_all_seq_len = []
    dfs = []
    seq_len = num_target_sets.index[0][0]

    for i in range(len(num_target_sets)):
        # randomly pair up contexts to target

        n = num_target_sets["count"].iloc[i]
        contexts_shuffled = contexts[i].sample(frac=1).iloc[: (n * NUM_CONTEXT)]

        df = targets[i].reset_index().rename(columns={"x": "target_x", "y": "target_y"})
        df["context_x"] = (
            contexts_shuffled["x"].to_numpy().reshape(n, NUM_CONTEXT).tolist()
        )
        df["context_y"] = (
            contexts_shuffled["y"].to_numpy().reshape(n, NUM_CONTEXT).tolist()
        )

        dfs.append(df)

        # TODO this is a mess
        if (
            i + 1 == len(num_target_sets)
            or num_target_sets.index[i][0] != num_target_sets.index[i + 1][0]
        ):
            # print(num_target_sets.index[i][0])
            dfs = pd.concat(dfs)
            dfs = dfs.sample(n=(len(dfs) // BATCH_SIZE) * BATCH_SIZE)
            dfs_all_seq_len.append(dfs)
            dfs = []

    dfs_all_seq_len = pd.concat(dfs_all_seq_len).reset_index()

    shuffled = (
        dfs_all_seq_len.assign(group_var=dfs_all_seq_len.index // BATCH_SIZE)
        .set_index("group_var")
        .loc[np.random.permutation(len(dfs_all_seq_len) // BATCH_SIZE)]
    )

    shuffled["context_x"] = shuffled["context_x"].map(lambda x: torch.cat(x, dim=3))
    shuffled["context_y"] = shuffled["context_y"].map(lambda x: torch.cat(x, dim=3))

    for start in range(0, len(shuffled), BATCH_SIZE):
        batch = shuffled.iloc[start : start + BATCH_SIZE]

        x_context = torch.cat(batch["context_x"].tolist()).to(device)
        y_context = torch.cat(batch["context_y"].tolist()).to(device)
        x_target = torch.cat(batch["target_x"].tolist()).to(device)
        y_target = torch.cat(batch["target_y"].tolist()).to(device)

        y_context_pos = torch.clip(
            1
            / (
                y_context + 1e-6 * (torch.randn(*y_context.shape).to(device) - 0.5)
            ),  # add some jitter
            min=-10.0,
            max=10.0,
        )

        y_target_pos = torch.clip(
            1
            / (
                y_target + 1e-6 * (torch.randn(*y_target.shape).to(device) - 0.5)
            ),  # add some jitter
            min=-10.0,
            max=10.0,
        )

        pred_y, _, train_loss, kl = model.forward(
            x_context,
            y_context_pos,
            x_target,
            y_target_pos,
        )

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # train_mse.append(
        #     (torch.mean((pred_y[:, :, :] - y_target[:, -1:, :]) ** 2)).detach().item()
        # )
        captured_return.append(
            (torch.mean(pred_y[:, :, :] * y_target[:, -1:, :]))
            .detach()
            .item()
        )
        # print(np.mean(train_mse))
        # break

    # print("Train MSE: ", np.mean(train_mse))
    print(
        "Train Sharpe Indv: ",
        np.mean(captured_return) / np.std(captured_return) * np.sqrt(252),
    )

    # TESTING
    test_results = (
        test_data_prepped_all_segments[[]]
        .copy()
        .assign(captured_return=0.0)
        # .assign(mse=0.0)
        .sort_index()
    )

    for key, test in test_data_prepped_all_segments.iterrows():
        seq_len, ticker, date = key
        context_points = train_data_prepped_all_segments.loc[(seq_len, ticker)].sample(
            n=NUM_CONTEXT
        )

        x_context = torch.cat(context_points["x"].tolist(), axis=3).to(device)
        y_context = torch.cat(context_points["y"].tolist(), axis=3).to(device)
        x_target = test["x"].to(device)
        y_target = test["y"].to(device)

        y_context_pos = torch.clip(
            1
            / (
                y_context + 1e-6 * (torch.randn(*y_context.shape).to(device) - 0.5)
            ),  # add some jitter
            min=-10.0,
            max=10.0,
        )

        y_target_pos = torch.clip(
            1
            / (
                y_target + 1e-6 * (torch.randn(*y_target.shape).to(device) - 0.5)
            ),  # add some jitter
            min=-10.0,
            max=10.0,
        )

        pred_y_test, _, test_loss, _ = model.forward(
            x_context, y_context_pos, x_target, y_target_pos, testing=True
        )

        # test_results.loc[key, "mse"] = (
        #     (torch.mean((pred_y_test[:, :, :] - y_target[:, -1:, :]) ** 2))
        #     .detach()
        #     .item()
        # )

        test_results.loc[key, "captured_return"] = (
            (torch.mean(pred_y_test[:, :, :] * y_target[:, -1:, :]))
            .detach()
            .item()
        )
        # break

    test_results = test_results.reset_index()
    valid_results = test_results[
        test_results["date"] < dt.datetime(TEST_START_YEAR, 1, 1)
    ]
    test_results = test_results[
        test_results["date"] >= dt.datetime(TEST_START_YEAR, 1, 1)
    ]

    valid_results_port = valid_results.groupby("date")["captured_return"].sum() / N
    valid_sharpe = (
        np.mean(valid_results_port) / np.std(valid_results_port) * np.sqrt(252)
    )
    print("Valid Sharpe Port: ", valid_sharpe)

    test_results_port = test_results.groupby("date")["captured_return"].sum() / N
    test_sharpe = np.mean(test_results_port) / np.std(test_results_port) * np.sqrt(252)
    print("Test Sharpe Port: ", test_sharpe)

    if valid_sharpe >= best_valid_sharpe:
        best_valid_sharpe = valid_sharpe

        if not os.path.exists("results"):
            os.mkdir("results")
        valid_results.to_csv(os.path.join("results", RUN_NAME + "_valid.csv"))
        test_results.to_csv(os.path.join("results", RUN_NAME + "_test.csv"))
        with open(
            os.path.join("results", RUN_NAME + "_results.json"), "w", encoding="utf-8"
        ) as f:
            # TODO other settings results
            json.dump(
                {
                    "valid_sharpe": valid_sharpe,
                    "test_sharpe": test_sharpe,
                    "iteration": it,
                },
                f,
                indent=4,
            )

    # if batch["x"].iloc[0].shape[1] == 3:
    #     break
