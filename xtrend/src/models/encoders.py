import torch
from torch import nn
import torch.nn.functional as F

from models.attention import Attention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DeterministicLSTMEncoder(nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim,
        hidden_dim_list,  # the dims of hidden starts of mlps
        embedding_dim=32,  # the dim of last axis of r..
        context_concat='stack',
        self_attention_type="dot",
        use_self_attn=False,
        use_x_attn=False,
        cross_attention_type="dot",
    ):
        super().__init__()
        self.concat = context_concat
        # stacking x and y and encoding
        if self.concat == 'stack':
            self.input_dim = x_dim + y_dim
        # appending y to x and encoding
        elif self.concat == 'append':
            self.input_dim = x_dim
        else:
            raise ValueError
        self.hidden_dim_list = hidden_dim_list
        self.hidden_dim = hidden_dim_list[-1]
        self.embedding_dim = embedding_dim
        self.use_self_attn = use_self_attn
        self.use_x_attn = use_x_attn

        self.num_rnn_layers = 1
        self.rnn           = nn.LSTM(self.input_dim, self.hidden_dim, self.num_rnn_layers)
        self.rnn_key_enc   = nn.LSTM(x_dim, self.hidden_dim, self.num_rnn_layers)
        self.rnn_query_enc = nn.LSTM(x_dim, self.hidden_dim, self.num_rnn_layers)

        if embedding_dim != hidden_dim_list[-1]:
            print('Warning, Check the dim of latent z and the dim of mlp last layer!')

        if self.use_self_attn:
            self._self_attention = Attention(
                self.hidden_dim,
                attention_type=self_attention_type,
                rep='', # can use mlp, have already encoded sequences using an LSTM so not needed
            )

        if self.use_x_attn:
            self._cross_attention = Attention(
                hidden_dim=self.hidden_dim,
                attention_type=cross_attention_type,
                x_dim=x_dim,
                rep='', # can use mlp, have already encoded sequences using an LSTM so not needed
            )

    def forward(self, context_x, context_y, target_x=None):
        batch_sz, context_len, y_dim, num_contexts = context_x.size()  # [batch_size, seq_len, y_size, num_contexts]

        if self.use_x_attn:
            # Encode the context_x and context_y as Values
            # Each time-series in the context is encoded such that a single
            # hidden state represents an encoding of the context
            _, target_len, _ = target_x.size()  # [batch_size, target_len, y_size]

            hidden_v = torch.zeros(batch_sz, num_contexts, self.hidden_dim).to(device)
            for i in range(num_contexts):
                if self.concat == 'stack':
                    h0 = torch.randn(self.num_rnn_layers, context_len, self.hidden_dim).to(device)  # hidden states
                    c0 = torch.randn(self.num_rnn_layers, context_len, self.hidden_dim).to(device)  # cell states
                    encoder_input = torch.cat([context_x[:, :, :, i], context_y[:, :, :, i]],
                                              dim=-1)  # (b, seq_len, 2 * y_dim)
                elif self.concat == 'append':
                    h0 = torch.randn(self.num_rnn_layers, 2 * context_len, self.hidden_dim).to(device)  # hidden states
                    c0 = torch.randn(self.num_rnn_layers, 2 * context_len, self.hidden_dim).to(device)  # cell states
                    encoder_input = torch.cat([context_x[:, :, :, i], context_y[:, :, :, i]],
                                              dim=1)  # (b, 2 * seq_len, y_dim)
                else:
                    raise ValueError

                hidden_r_i, _ = self.rnn(encoder_input, (h0, c0))  # (b, context_seq_len, latent_dim)

                hidden_v[:, i, :] = hidden_r_i[:, -1, :]  # (b, hidden_dim)

            # self attention over values
            if self.use_self_attn:
                hidden_v = self._self_attention(hidden_v, hidden_v, hidden_v)

            # Encode the context_x as Keys
            h0 = torch.randn(self.num_rnn_layers, context_len, self.hidden_dim).to(device)
            c0 = torch.randn(self.num_rnn_layers, context_len, self.hidden_dim).to(device)
            hidden_k = torch.zeros(batch_sz, num_contexts, self.hidden_dim).to(device)
            for i in range(num_contexts):
                hidden_r_i, _ = self.rnn_key_enc(context_x[:, :, :, i], (h0, c0))
                hidden_k[:, i, :] = hidden_r_i[:, -1, :]

            # Encoder the target_x as Query
            h0 = torch.randn(self.num_rnn_layers, target_len, self.hidden_dim).to(device)
            c0 = torch.randn(self.num_rnn_layers, target_len, self.hidden_dim).to(device)
            hidden_r_i, _ = self.rnn_query_enc(target_x, (h0, c0)) # (b, num_contexts, hidden_dim)
            hidden_q = hidden_r_i[:, [-1], :]

            # hidden_k: (b, num_contexts, hidden_dim)
            # hidden_v: (b, num_contexts, hidden_dim)
            # hidden_q: (b, 1, hidden_dim)
            representation = self._cross_attention(hidden_k, hidden_v, hidden_q)  # (b, 1, latent_dim)
            representation = representation.squeeze(1)
        else:
            representation = torch.zeros(batch_sz, self.hidden_dim).to(device)
            for i in range(num_contexts):
                if self.concat == 'stack':
                    h0 = torch.randn(self.num_rnn_layers, context_len, self.hidden_dim).to(device)  # hidden states
                    c0 = torch.randn(self.num_rnn_layers, context_len, self.hidden_dim).to(device)  # cell states
                    encoder_input = torch.cat([context_x[:, :, :, i], context_y[:, :, :, i]], dim=-1) # (b, seq_len, 2 * y_dim)
                elif self.concat == 'append':
                    h0 = torch.randn(self.num_rnn_layers, 2 * context_len, self.hidden_dim).to(device)  # hidden states
                    c0 = torch.randn(self.num_rnn_layers, 2 * context_len, self.hidden_dim).to(device)  # cell states
                    encoder_input = torch.cat([context_x[:, :, :, i], context_y[:, :, :, i]], dim=1)  # (b, 2 * seq_len, y_dim)
                else:
                    raise ValueError

                hidden_r_i, _ = self.rnn(encoder_input, (h0, c0)) # (b, context_seq_len, latent_dim)

                representation += hidden_r_i[:, -1, :]  # (b, hidden_dim)
        return representation


class LatentLSTMEncoder(nn.Module):
    def __init__(self,
                 x_dim,
                 y_dim,
                 hidden_dim,
                 latent_dim,
                 context_concat='stack',
                 use_self_attn=False,
                 self_attention_type="dot"):

        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.concat = context_concat
        if self.concat == 'stack':
            self.input_dim = x_dim + y_dim
        # appending y to x and encoding
        elif self.concat == 'append':
            self.input_dim = x_dim
        else:
            raise ValueError
        self.num_rnn_layers = 1
        self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, self.num_rnn_layers)
        self.mean_transform = nn.Linear(self.hidden_dim, self.latent_dim)
        self.log_var_transform = nn.Linear(self.hidden_dim, self.latent_dim)
        self.use_self_attn = use_self_attn
        #self.self_attention = Attention(n_units, self_attention_type)

    def forward(self, context_x, context_y, state=None):
        batch_sz, context_x_len, y_dim, num_contexts = context_x.size()  # [batch_size, seq_len, y_size, num_contexts]
        _, context_y_len, _, _ = context_y.size()  # [batch_size, seq_len, y_size, num_contexts]

        representation = torch.zeros(batch_sz, self.hidden_dim).to(device)
        for i in range(num_contexts):
            if self.concat == 'stack':
                if state is None:
                    assert context_x_len == context_y_len
                    h0 = torch.randn(self.num_rnn_layers, context_x_len, self.hidden_dim).to(device)  # hidden states
                    c0 = torch.randn(self.num_rnn_layers, context_x_len, self.hidden_dim).to(device)  # cell states
                    state = (h0, c0)
                encoder_input = torch.cat([context_x[:, :, :, i], context_y[:, :, :, i]],
                                          dim=-1)  # (b, seq_len, 2 * y_dim)
            elif self.concat == 'append':
                if state is None:
                    h0 = torch.randn(self.num_rnn_layers, context_x_len + context_y_len, self.hidden_dim).to(device)  # hidden states
                    c0 = torch.randn(self.num_rnn_layers, context_x_len + context_y_len, self.hidden_dim).to(device)  # cell states
                    state = (h0, c0)
                encoder_input = torch.cat([context_x[:, :, :, i], context_y[:, :, :, i]],
                                          dim=1)  # (b, 2 * seq_len, y_dim)
            else:
                raise ValueError

            hidden_r_i, _ = self.rnn(encoder_input, state)  # (b, context_seq_len, latent_dim)

            if self.use_self_attn:
                pass

            representation += (hidden_r_i[:, -1, :] / num_contexts) # (b, hidden_dim)

        #r = self.self_attention(r, r, r)
        mu = self.mean_transform(representation)
        log_var = self.log_var_transform(representation)

        sigma = 0.1 + 0.9 * F.softplus(0.5 * log_var)

        dist = torch.distributions.Normal(mu, sigma)

        z = mu + sigma * torch.randn_like(mu)
        return z, dist, state
