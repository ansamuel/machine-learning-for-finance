from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
import datetime as dt
import torch
import random

# from rnn_np import ANP_RNN_Model
import os
import json

import torch
from torch import nn
import torch.nn.functional as F

from models.attention import Attention

TESTING = False
TRAIN_LAST_OUTPUT = 0  # more items in sequence is neccessary!
LATENT_PATH = False
VAL_DIVERSIFIED_SHARPE = True


BATCH_SIZE = 64
# NUM_CONTEXT = 10
NUM_CONTEXT = 10
ITERATIONS = 50
MIN_SEQ_LEN = 5

N = 50

LR = 0.001

# LEN_CONTEXT = 6
# BATCH_SIZE = 256

# EMBEDDING_DIM = 32
EMBEDDING_DIM = 16

ENCODER_OUTPUT_SIZES = [EMBEDDING_DIM, EMBEDDING_DIM]
DECODER_OUTPUT_SIZES = [EMBEDDING_DIM, EMBEDDING_DIM]

LATENT_DIM = 4
# LATENT_DIM = 4
USE_X_ATTENTION = True
USE_SELF_ATTENTION = True

VALID_START_YEAR = 2015  # valid 2015-208

TEST_START_YEAR = 2020  # valid 2015-208


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if TESTING:
    all_segments = pd.read_pickle("data/all_data_AN.pkl")
else:
    all_segments = pd.read_pickle("data/all_data.pkl")

all_segments = all_segments[all_segments["seq_len"] >= MIN_SEQ_LEN].copy()

all_segments.index = all_segments.groupby(["seq_len", "ticker"]).cumcount()

all_segments["day"] = all_segments.groupby("date").ngroup()
# print(all_segments)

all_segments_day = all_segments.set_index("day")
all_segments_day_dict = {}
for s in all_segments_day.seq_len.unique():
    all_segments_day_dict[s] = all_segments_day[all_segments_day["seq_len"] == s]

targets = all_segments[all_segments.index > NUM_CONTEXT]

train_targets = targets[targets.date.dt.year < VALID_START_YEAR]
valid_targets = targets[
    (targets.date.dt.year >= VALID_START_YEAR)
    & (targets.date.dt.year < TEST_START_YEAR)
]


def prepare_batches(targets, keep_partial_batches = False):
    # shuffled = train_targets.sample(frac=1)
    shuffled = targets.sample(frac=1)
    shuffled["batch"] = shuffled.groupby("seq_len").cumcount() // BATCH_SIZE
    grouped = shuffled.groupby(["seq_len", "batch"])
    if not keep_partial_batches:
        shuffled = grouped.filter(lambda x: x["batch"].count() == BATCH_SIZE)
    shuffled.index = shuffled.groupby(["seq_len", "batch"]).ngroup()
    num_batches = shuffled.index.max()
    order = list(range(num_batches))
    random.shuffle(order)
    # print(order)
    # print(shuffled.loc[order])

    shuffled["contexts"] = shuffled.apply(
        lambda row: all_segments_day_dict[row.seq_len]
        # .loc[: (row.day - 1)]
        .loc[lambda df: df.index < row.day][["x", "y"]].sample(n=NUM_CONTEXT),
        axis=1,
    )

    shuffled["context_x"] = shuffled["contexts"].map(
        lambda c: torch.stack(c["x"].tolist(), dim=3)
    )
    shuffled["context_y"] = shuffled["contexts"].map(
        lambda c: torch.stack(c["y"].tolist(), dim=3)
    )

    # print(shuffled["context_x"].iloc[0].shape)
    batches = []
    for i in order:
        batch = shuffled.loc[[i]]
        # print(batch)
        # print(torch.cat(batch["context_x"].tolist()).shape)
        batches.append(
            (
                batch["seq_len"].iloc[0],
                torch.cat(batch["context_x"].tolist()),
                torch.cat(batch["context_y"].tolist()),
                torch.cat(batch["x"].tolist()),
                torch.cat(batch["y"].tolist()),
                batch["date"].tolist(),
                batch["ticker"].tolist(),
            )
        )

    # print(shuffled["context_x"].iloc[0].shape)
    return batches


# prepare_batches(train_targets)[0]


class DeterministicLSTMEncoder(nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim,
        hidden_dim_list,  # the dims of hidden starts of mlps
        embedding_dim=32,  # the dim of last axis of r..
        context_concat="stack",
        self_attention_type="dot",
        use_self_attn=False,
        use_x_attn=False,
        cross_attention_type="dot",
    ):
        super().__init__()
        self.concat = context_concat
        # stacking x and y and encoding
        if self.concat == "stack":
            self.input_dim = x_dim + y_dim
        # appending y to x and encoding
        elif self.concat == "append":
            self.input_dim = x_dim
        else:
            raise ValueError
        self.hidden_dim_list = hidden_dim_list
        self.hidden_dim = hidden_dim_list[-1]
        self.embedding_dim = embedding_dim
        self.use_self_attn = use_self_attn
        self.use_x_attn = use_x_attn

        self.num_rnn_layers = 1
        self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, self.num_rnn_layers)
        self.rnn_key_enc = nn.LSTM(x_dim, self.hidden_dim, self.num_rnn_layers)
        self.rnn_query_enc = nn.LSTM(x_dim, self.hidden_dim, self.num_rnn_layers)

        if embedding_dim != hidden_dim_list[-1]:
            print("Warning, Check the dim of latent z and the dim of mlp last layer!")

        if self.use_self_attn:
            self._self_attention = Attention(
                self.hidden_dim,
                attention_type=self_attention_type,
                rep="",  # can use mlp, have already encoded sequences using an LSTM so not needed
            )

        if self.use_x_attn:
            self._cross_attention = Attention(
                hidden_dim=self.hidden_dim,
                attention_type=cross_attention_type,
                x_dim=x_dim,
                rep="",  # can use mlp, have already encoded sequences using an LSTM so not needed
            )

    def forward(self, context_x, context_y, target_x=None):
        (
            batch_sz,
            context_len,
            y_dim,
            num_contexts,
        ) = context_x.size()  # [batch_size, seq_len, y_size, num_contexts]

        if self.use_x_attn:
            # Encode the context_x and context_y as Values
            # Each time-series in the context is encoded such that a single
            # hidden state represents an encoding of the context
            _, target_len, _ = target_x.size()  # [batch_size, target_len, y_size]

            hidden_v = torch.zeros(
                batch_sz, context_len, num_contexts, self.hidden_dim
            ).to(device)
            for i in range(num_contexts):
                if self.concat == "stack":
                    # h0 = torch.randn(
                    #     self.num_rnn_layers, context_len, self.hidden_dim
                    # ).to(
                    #     device
                    # )  # hidden states
                    # c0 = torch.randn(
                    #     self.num_rnn_layers, context_len, self.hidden_dim
                    # ).to(
                    #     device
                    # )  # cell states
                    encoder_input = torch.cat(
                        [context_x[:, :, :, i], context_y[:, :, :, i]], dim=-1
                    )  # (b, seq_len, 2 * y_dim)
                elif self.concat == "append":
                    # h0 = torch.randn(
                    #     self.num_rnn_layers, 2 * context_len, self.hidden_dim
                    # ).to(
                    #     device
                    # )  # hidden states
                    # c0 = torch.randn(
                    #     self.num_rnn_layers, 2 * context_len, self.hidden_dim
                    # ).to(
                    #     device
                    # )  # cell states
                    encoder_input = torch.cat(
                        [context_x[:, :, :, i], context_y[:, :, :, i]], dim=1
                    )  # (b, 2 * seq_len, y_dim)
                else:
                    raise ValueError

                hidden_r_i, _ = self.rnn(
                    # encoder_input, (h0, c0)
                    encoder_input
                )  # (b, context_seq_len, latent_dim)

                hidden_v[:, :, i, :] = hidden_r_i  # (b, hidden_dim)

            # added to do self attention for each timestep
            shape = hidden_v.shape
            hidden_v = hidden_v.reshape((shape[0] * shape[1], shape[2], shape[3]))

            # self attention over values
            if self.use_self_attn:
                hidden_v = self._self_attention(hidden_v, hidden_v, hidden_v)

            # KW maybe just keep hidden and cell as 0 for now...
            # # Encode the context_x as Keys
            # h0 = torch.randn(self.num_rnn_layers, context_len, self.hidden_dim).to(
            #     device
            # )
            # c0 = torch.randn(self.num_rnn_layers, context_len, self.hidden_dim).to(
            #     device
            # )

            hidden_k = torch.zeros(
                batch_sz, context_len, num_contexts, self.hidden_dim
            ).to(device)
            for i in range(num_contexts):
                # hidden_r_i, _ = self.rnn_key_enc(context_x[:, :, :, i], (h0, c0))
                hidden_k_i, _ = self.rnn_key_enc(context_x[:, :, :, i])
                hidden_k[:, :, i, :] = hidden_k_i

            hidden_k = hidden_k.reshape((shape[0] * shape[1], shape[2], shape[3]))

            # set for c0, h0 to 0 for now...
            # # Encoder the target_x as Query
            # h0 = torch.randn(self.num_rnn_layers, target_len, self.hidden_dim).to(
            #     device
            # )
            # c0 = torch.randn(self.num_rnn_layers, target_len, self.hidden_dim).to(
            #     device
            # )
            # hidden_r_i, _ = self.rnn_query_enc(
            #     target_x, (h0, c0)
            # )  # (b, num_contexts, hidden_dim)
            hidden_q, _ = self.rnn_query_enc(target_x)

            shape_q = hidden_q.shape
            hidden_q = hidden_q.reshape(shape_q[0] * shape_q[1], 1, shape_q[2])

            # hidden_k: (b, num_contexts, hidden_dim)
            # hidden_v: (b, num_contexts, hidden_dim)
            # hidden_q: (b, 1, hidden_dim)
            representation = self._cross_attention(
                hidden_k, hidden_v, hidden_q
            ).reshape(shape_q)

        else:
            # TODO only done X attention path so far
            representation = torch.zeros(batch_sz, self.hidden_dim).to(device)
            for i in range(num_contexts):
                if self.concat == "stack":
                    h0 = torch.randn(
                        self.num_rnn_layers, context_len, self.hidden_dim
                    ).to(
                        device
                    )  # hidden states
                    c0 = torch.randn(
                        self.num_rnn_layers, context_len, self.hidden_dim
                    ).to(
                        device
                    )  # cell states
                    encoder_input = torch.cat(
                        [context_x[:, :, :, i], context_y[:, :, :, i]], dim=-1
                    )  # (b, seq_len, 2 * y_dim)
                elif self.concat == "append":
                    h0 = torch.randn(
                        self.num_rnn_layers, 2 * context_len, self.hidden_dim
                    ).to(
                        device
                    )  # hidden states
                    c0 = torch.randn(
                        self.num_rnn_layers, 2 * context_len, self.hidden_dim
                    ).to(
                        device
                    )  # cell states
                    encoder_input = torch.cat(
                        [context_x[:, :, :, i], context_y[:, :, :, i]], dim=1
                    )  # (b, 2 * seq_len, y_dim)
                else:
                    raise ValueError

                hidden_r_i, _ = self.rnn(
                    encoder_input, (h0, c0)
                )  # (b, context_seq_len, latent_dim)

                representation += hidden_r_i[:, -1, :]  # (b, hidden_dim)
        return representation


class LatentLSTMEncoder(nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim,
        hidden_dim,
        latent_dim,
        context_concat="stack",
        use_self_attn=False,
        self_attention_type="dot",
    ):

        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.concat = context_concat
        if self.concat == "stack":
            self.input_dim = x_dim + y_dim
        # appending y to x and encoding
        elif self.concat == "append":
            self.input_dim = x_dim
        else:
            raise ValueError
        self.num_rnn_layers = 1
        self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, self.num_rnn_layers)
        self.mean_transform = nn.Linear(self.hidden_dim, self.latent_dim)
        self.log_var_transform = nn.Linear(self.hidden_dim, self.latent_dim)
        self.use_self_attn = use_self_attn
        # self.self_attention = Attention(n_units, self_attention_type)

    def forward(self, context_x, context_y, state=None):

        (
            batch_sz,
            context_x_len,
            y_dim,
            num_contexts,
        ) = context_x.size()  # [batch_size, seq_len, y_size, num_contexts]
        (
            _,
            context_y_len,
            _,
            _,
        ) = context_y.size()  # [batch_size, seq_len, y_size, num_contexts]

        representation = torch.zeros(batch_sz, context_x_len, self.hidden_dim).to(
            device
        )
        for i in range(num_contexts):
            if self.concat == "stack":
                if state is None:
                    assert context_x_len == context_y_len
                    # h0 = torch.randn(self.num_rnn_layers, context_x_len, self.hidden_dim).to(device)  # hidden states
                    # c0 = torch.randn(self.num_rnn_layers, context_x_len, self.hidden_dim).to(device)  # cell states
                    # state = (h0, c0)
                encoder_input = torch.cat(
                    [context_x[:, :, :, i], context_y[:, :, :, i]], dim=-1
                )  # (b, seq_len, 2 * y_dim)
            elif self.concat == "append":
                # if state is None:
                # h0 = torch.randn(self.num_rnn_layers, context_x_len + context_y_len, self.hidden_dim).to(device)  # hidden states
                # c0 = torch.randn(self.num_rnn_layers, context_x_len + context_y_len, self.hidden_dim).to(device)  # cell states
                # state = (h0, c0)
                encoder_input = torch.cat(
                    [context_x[:, :, :, i], context_y[:, :, :, i]], dim=1
                )  # (b, 2 * seq_len, y_dim)
            else:
                raise ValueError

            hidden_r_i, _ = self.rnn(encoder_input)  # (b, context_seq_len, latent_dim)

            if self.use_self_attn:
                pass

            # average
            # TODO can remove loop
            representation += hidden_r_i / num_contexts  # (b, hidden_dim)

        # r = self.self_attention(r, r, r)
        mu = self.mean_transform(representation)
        log_var = self.log_var_transform(representation)

        sigma = 0.1 + 0.9 * F.softplus(0.5 * log_var)

        dist = torch.distributions.Normal(mu, sigma)

        z = mu + sigma * torch.randn_like(mu)
        return z, dist, state


class DeterministicLSTMDecoder(nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim,
        hidden_dim_list,  # the dims of hidden starts of mlps
        embedding_dim,  # the dim of last axis of x, r and z..
        x_attn_repr=False,
    ):
        """

        :params x_dim: int
        :params y_dim: int
        :params hidden_dim_list: list
        :params embedding_dim: int
        """
        super(DeterministicLSTMDecoder, self).__init__()

        self.hidden_dim_list = hidden_dim_list
        self.hidden_dim = hidden_dim_list[-1]
        self.input_dim = embedding_dim + x_dim
        self.x_attn_repr = x_attn_repr
        self.rnn = nn.LSTMCell(self.input_dim, self.hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, y_dim),
            nn.Tanh(),
        )

    def forward(
        self, r, target_x, target_y, z=None, testing=False, teacher_forcing=False
    ):
        # r:        (b, hidden_dim)
        # z:        (b, latent_dim)
        # target_x: (b, target_x_seq_len, x_dim)
        # target_y: (b, target_y_seq_len, x_dim)
        batch_size, len_target_x, _ = target_x.size()
        _, len_target_y, _ = target_y.size()

        # h = torch.randn(batch_size, self.hidden_dim).to(device)  # hidden states
        # c = torch.randn(batch_size, self.hidden_dim).to(device)  # cell states

        # concatenate target_x and representation
        # r_tiled = torch.tile(
        #     r.unsqueeze(1), (1, len_target_x, 1)
        # )  # (b, target_len, hidden_dim)
        if z is not None:
            # z_tiled = torch.tile(
            #     z.unsqueeze(1), (1, len_target_x, 1)
            # )  # (b, target_len, hidden_dim)
            decoder_input = torch.cat(
                [z, r, target_x], dim=-1
            )  # (b, target_len, latent_dim + hidden_dim + x_dim)
        else:
            decoder_input = torch.cat(
                [r, target_x], dim=-1
            )  # (b, target_len, hidden_dim + x_dim)

        # encode the target sequence

        # TODO think we can do this without looping

        # h = torch.randn(batch_size, self.hidden_dim).to(device)  # hidden states
        # c = torch.randn(batch_size, self.hidden_dim).to(device)  # cell states

        h = torch.zeros((batch_size, self.hidden_dim)).to(device)  # hidden states
        c = torch.zeros((batch_size, self.hidden_dim)).to(device)  # cell states

        h_vector = torch.zeros(batch_size, len_target_x, self.hidden_dim).to(device)

        h_list = []
        for i in range(len_target_x):
            h, c = self.rnn(decoder_input[:, i, :], (h, c))  # (b, hidden_dim)
            h_list.append(h)

        h_vector = torch.stack(h_list).swapaxes(0, 1)

        # mu_sigma = self.out(h) # final prediction only (b, 2 * y_dim)

        # mu, log_sigma = mu_sigma.chunk(chunks=2, dim=-1) # mu/sigma (b, 1)

        # # # Bound the variance
        # sigma = 0.1 + 0.9 * F.softplus(log_sigma)

        # # Get the distribution
        # dist = torch.distributions.Normal(mu, sigma)

        positions = self.out(h_vector)  # final prediction only (b, 2 * y_dim)

        captured_positions = y_target * positions

        if testing:
            return captured_positions

        if TRAIN_LAST_OUTPUT:
            captured_positions = captured_positions[:, -TRAIN_LAST_OUTPUT:, :]

        # TODO replace with a tensor sqrt
        # sharpe = torch.mean(
        #     torch.mean(captured_positions, dim=1)
        #     / (torch.std(captured_positions, dim=1) + 1e-9)
        # ) * np.sqrt(252.0)
        sharpe = (
            torch.mean(captured_positions)
            / (torch.std(captured_positions) + 1e-9)
        ) * np.sqrt(252.0)
        return -sharpe


class ANP_RNN_Model(nn.Module):
    """
    (Attentive) Neural Process model
    https://github.com/VersElectronics/Neural-Processes/blob/master/neural_process_models/anp_rnn.py
    """

    def __init__(
        self,
        x_dim,
        y_dim,
        encoder_rnn_hidden_size_list,
        decoder_rnn_hidden_size_list,
        embedding_dim,
        latent_dim,
        context_concat="stack",
        use_self_attention=False,
        use_cross_attention=False,
        le_self_attention_type="dot",
        de_self_attention_type="dot",
        de_cross_attention_type="multihead",
        latent_path=False,
    ):
        """
        :params x_dim: int
        :params y_dim: int
        :params encoder_rnn_hidden_size_list: list
        :params decoder_rnn_hidden_size_list: list
        :params embedding_dim: int
        """
        super(ANP_RNN_Model, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.encoder_rnn_hidden_size_list = encoder_rnn_hidden_size_list
        self.decoder_rnn_hidden_size_list = decoder_rnn_hidden_size_list
        self.embedding_dim = embedding_dim
        self.num_rnn_layers = 1
        self.use_cross_attention = use_cross_attention
        self.latent_dim = latent_dim
        self.latent_path = latent_path

        self._deter_encoder = DeterministicLSTMEncoder(
            x_dim=self.x_dim,
            y_dim=self.y_dim,
            hidden_dim_list=self.encoder_rnn_hidden_size_list,
            embedding_dim=self.embedding_dim,  # the dim of last axis of r..
            context_concat=context_concat,
            use_x_attn=use_cross_attention,
            cross_attention_type=de_cross_attention_type,
            use_self_attn=use_self_attention,
            self_attention_type=de_self_attention_type,
        )

        self._decoder = DeterministicLSTMDecoder(
            x_dim=self.x_dim,
            y_dim=self.y_dim,
            hidden_dim_list=self.decoder_rnn_hidden_size_list,
            embedding_dim=self.embedding_dim + self.latent_dim
            if self.latent_path
            else self.embedding_dim,
            x_attn_repr=use_cross_attention,
        )

        if self.latent_path:
            self._lat_encoder = LatentLSTMEncoder(
                x_dim=self.x_dim,
                y_dim=self.y_dim,
                hidden_dim=self.encoder_rnn_hidden_size_list[-1],
                latent_dim=self.latent_dim,
                context_concat=context_concat,
            )

    def forward(self, context_x, context_y, target_x, target_y, testing=False):
        # If testing is True we unroll the LSTM and use the previous timestep predictions
        # as an input for a new prediction.

        r = self._deter_encoder(
            context_x, context_y, target_x if self.use_cross_attention else None
        )

        if self.latent_path:
            # TODO pass state from latent encoder context to latent encoding of the targets
            z, prior_dist, _ = self._lat_encoder(
                context_x, context_y
            )  # z (b, latent_dim)
            z_post, post_dist, _ = self._lat_encoder(
                target_x.unsqueeze(-1), target_y.unsqueeze(-1)
            )
        else:
            z = None
        # If we want to calculate the log_prob for training we will make use of the
        # target_y. At test time the target_y is not available so we return None.
        batch_size, len_target_y, _ = target_y.size()

        loss_or_pos_testing = self._decoder(r, target_x, target_y, z, testing=testing)

        return loss_or_pos_testing


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
    latent_path=LATENT_PATH,
)

model = model.to(device)

# Set up the optimizer and train step
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# probably need to separate the target data out earlier

best_valid_sharpe = 0
epoch_sharpes = []
for it in range(ITERATIONS):
    print(f"Iteration {it}")

    train_sharpes = []
    # train_mse = []

    print("Prepping")
    batches = prepare_batches(train_targets)
    print("Ready")
    for seq_len, x_context, y_context, x_target, y_target, _, _ in batches:

        x_context = x_context.to(device)
        y_context = y_context.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)

        train_loss = model.forward(
            x_context,
            y_context,
            x_target,
            y_target,
        )
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # train_mse.append(
        #     (torch.mean((pred_y[:, :, :] - y_target[:, -1:, :]) ** 2)).detach().item()
        # )
        train_sharpes.append((-train_loss).detach().item())
        if TESTING:
            break

    # print(train_sharpes)
    # epoch_sharpes.append(np.mean(sharpes))
    print("Train Sharpe: ", np.mean(train_sharpes))

    valid_sharpes = []

    # train_mse = []

    all_returns = []
    targ = valid_targets if not TESTING else train_targets
    for (
        seq_len,
        x_context,
        y_context,
        x_target,
        y_target,
        dates,
        tickers,
    ) in prepare_batches(targ, keep_partial_batches=True):
        x_context = x_context.to(device)
        y_context = y_context.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)

        valid_loss = model.forward(
            x_context, y_context, x_target, y_target, VAL_DIVERSIFIED_SHARPE
        )

        # valid_loss.backward()

        # valid_mse.append(
        #     (torch.mean((pred_y[:, :, :] - y_target[:, -1:, :]) ** 2)).detach().item()
        # )

        if VAL_DIVERSIFIED_SHARPE:
            captured_returns = valid_loss[:, -1, 0].tolist()
            all_returns += zip(dates, tickers, captured_returns)
        else:
            valid_sharpes.append((-valid_loss).detach().item())

    # print(valid_sharpes)
    # epoch_sharpes.append(np.mean(sharpes))
    if VAL_DIVERSIFIED_SHARPE:
        diversified = pd.DataFrame(
            all_returns, columns=["date", "ticker", "captured_returns"]
        ).groupby("date")["captured_returns"].sum()
        val_sharpe = diversified.mean() * np.sqrt(252) / diversified.std()
        print("Valid Sharpe: ", val_sharpe)
    else:
        print("Valid Sharpe: ", np.mean(valid_sharpes))

    # # TESTING
    # test_results = (
    #     test_data_prepped_all_segments[[]]
    #     .copy()
    #     .assign(captured_return=0.0)
    #     # .assign(mse=0.0)
    #     # .assign(pred=0.0)
    #     # .assign(std=0.0)
    #     .sort_index()
    # )

    # for key, test in test_data_prepped_all_segments.iterrows():
    #     seq_len, ticker, date = key
    #     context_points = train_data_prepped_all_segments.loc[(seq_len, ticker)].sample(
    #         n=NUM_CONTEXT
    #     )

    #     x_context = torch.cat(context_points["x"].tolist(), axis=3).to(device)
    #     y_context = torch.cat(context_points["y"].tolist(), axis=3).to(device)
    #     x_target = test["x"].to(device)
    #     y_target = test["y"].to(device)

    #     captured_pos = model.forward(
    #         x_context, y_context, x_target, y_target, testing=True
    #     )

    #     # test_results.loc[key, "mse"] = (
    #     #     (torch.mean((pred_y_test[:, :, :] - y_target[:, -1:, :]) ** 2)).detach().item()
    #     # )

    #     # test_results.loc[key, "pred"] = (
    #     #     torch.mean(pred_y_test[:, :, :])
    #     #     .detach()
    #     #     .item()
    #     # )

    #     # test_results.loc[key, "std"] = (
    #     #             torch.mean(sigma[:, :, :])
    #     #             .detach()
    #     #             .item()
    #     #         )

    #     test_results.loc[key, "captured_return"] = (
    #         captured_pos[-1, -1, -1].detach().item()
    #     )
    #     # break

    # test_results = test_results.reset_index()
    # valid_results = test_results[
    #     test_results["date"] < dt.datetime(TEST_START_YEAR, 1, 1)
    # ]
    # # test_results = test_results[
    # #     test_results["date"] >= dt.datetime(TEST_START_YEAR, 1, 1)
    # # ]

    # valid_results_port = valid_results.groupby("date")["captured_return"].sum() / N
    # # print(test_results)
    # valid_sharpe = (
    #     np.mean(valid_results_port) / np.std(valid_results_port) * np.sqrt(252)
    # )
    # print("Valid Sharpe Port: ", valid_sharpe)
    # print()

    # # test_results_port = test_results.groupby("date")["captured_return"].sum() / N
    # # test_sharpe = np.mean(test_results_port) / np.std(test_results_port) * np.sqrt(252)
    # # print("Test Sharpe Port: ", test_sharpe)

    # # if valid_sharpe >= best_valid_sharpe:
    # #     best_valid_sharpe = valid_sharpe

    # #     if not os.path.exists("results"):
    # #         os.mkdir("results")
    # #     valid_results.to_csv(os.path.join("results", RUN_NAME + "_valid.csv"))
    # #     test_results.to_csv(os.path.join("results", RUN_NAME + "_test.csv"))
    # #     with open(os.path.join("results", RUN_NAME + "_results.json"), "w", encoding="utf-8") as f:
    # #         # TODO other settings results
    # #         json.dump(
    # #             {
    # #                 "valid_sharpe": valid_sharpe,
    # #                 "test_sharpe": test_sharpe,
    # #                 "iteration": it,
    # #             },
    # #             f,
    # #             indent=4,
    #         )
