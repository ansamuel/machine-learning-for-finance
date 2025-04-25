'''
Implementation of decoder for neural process.

Steps:
1. concatenate the target_x and latent variables r_star and z
2. Then pass them input a MLP
3. According to deepmind imp, then split the hidden to get mu and sigma
    Maybe using reparamerization trick will break something here

Can be attended with cross-attention *Whether to put the cross-attention in this encoder?

From the deepmind implementation
decoder_output_sizes = [HIDDEN_SIZE]*2 + [2] => decoder_hidden_dim_list[-1] = 2
Here 2 comes from y_dim * 2

The operation on latent variables should be completed outside the decoder
'''

import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        self.x_attn_repr=x_attn_repr
        self.rnn = nn.LSTMCell(self.input_dim, self.hidden_dim)
        self.out = nn.Sequential(
                nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * self.hidden_dim, 2 * y_dim),
            )

    def forward(self, r, target_x, target_y, z=None, unroll=False):
        # r:        (b, hidden_dim)
        # z:        (b, latent_dim)
        # target_x: (b, target_x_seq_len, x_dim)
        # target_y: (b, target_y_seq_len, x_dim)
        batch_size, len_target_x, _ = target_x.size()
        _, len_target_y, _ = target_y.size()

        h = torch.randn(batch_size, self.hidden_dim).to(device)  # hidden states
        c = torch.randn(batch_size, self.hidden_dim).to(device)  # cell states

        # concatenate target_x and representation
        r_tiled = torch.tile(r.unsqueeze(1), (1, len_target_x, 1)) # (b, target_len, hidden_dim)
        if z is not None:
            z_tiled = torch.tile(z.unsqueeze(1), (1, len_target_x, 1))  # (b, target_len, hidden_dim)
            decoder_input = torch.cat([z_tiled, r_tiled, target_x], dim=-1) # (b, target_len, latent_dim + hidden_dim + x_dim)
        else:
            decoder_input = torch.cat([r_tiled, target_x], dim=-1)  # (b, target_len, hidden_dim + x_dim)

        # encode the target sequence
        for i in range(len_target_x):
            h, c = self.rnn(decoder_input[:, i, :],  (h, c)) # (b, hidden_dim)

        mu_sigma = self.out(h) # final prediction only (b, 2 * y_dim)

        mu, log_sigma = mu_sigma.chunk(chunks=2, dim=-1) # mu/sigma (b, 1)

        # Bound the variance
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)

        # Get the distribution
        dist = torch.distributions.Normal(mu, sigma)
        dists, mus, sigmas = [dist], [mu.unsqueeze(1)], [sigma.unsqueeze(1)]

        # Unroll LSTM to make predictions
        for i in range(len_target_y):
            # Teacher forcing or LSTM unroll to make seq prediction
            x = target_y[:, i, :] if unroll else mu
            if z is not None:
                decoder_input = torch.cat([z, r, x], dim=-1)  # (b, embed_dim + latent_dim + x_dim)
            else:
                decoder_input = torch.cat([r, x], dim=-1)  # (b, latent_dim + x_dim)

            h, c = self.rnn(decoder_input, (h, c)) # h (b x hidden_dim)
            mu_sigma = self.out(h)  # (b, 2 * y_dim)

            mu, log_sigma = mu_sigma.chunk(chunks=2, dim=-1)  # mu/sigma (b, 1)

            sigma = 0.1 + 0.9 * F.softplus(log_sigma)

            dist = torch.distributions.Normal(mu, sigma)

            dists.append(dist)
            mus.append(mu.unsqueeze(1))
            sigmas.append(sigma.unsqueeze(1))
        return dists, mus, sigmas
