import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LSTMBaseline(nn.Module):
    """
    LSTM baseline model on targets only
    """
    def __init__(self,
                 x_dim,
                 y_dim,
                 hidden_dim_list,  # the dims of hidden starts of mlps
                 ):
        super(LSTMBaseline, self).__init__()
        self.input_dim = x_dim
        self.hidden_dim_list = hidden_dim_list
        self.hidden_dim = hidden_dim_list[-1]
        self.num_rnn_layers = 1
        self.rnn = nn.LSTMCell(self.input_dim, self.hidden_dim, self.num_rnn_layers)
        self.out = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * self.hidden_dim, y_dim),
        )

    def forward(self, context_x, context_y, target_x, target_y, unroll=False):
        # contexts are ignored by the LSTM baseline model.
        # Just the CNP/NP type models will condition on a context.
        # target_x: (b, target_x_seq_len, x_dim)
        # target_y: (b, target_y_seq_len, x_dim)
        batch_size, len_target_x, _ = target_x.size()
        _, len_target_y, _ = target_y.size()

        h = torch.randn(batch_size, self.hidden_dim).to(device)  # hidden states
        c = torch.randn(batch_size, self.hidden_dim).to(device)  # cell states

        # encode the target sequence
        for i in range(len_target_x):
            h, c = self.rnn(target_x[:, i, :], (h, c))  # (b, hidden_dim)

        mu = self.out(h)  # final prediction only (b, y_dim)
        mus = [mu.unsqueeze(1)]
        loss, kl_loss = 0, 0
        # Unroll LSTM to make predictions
        for i in range(len_target_y):
            # Teacher forcing or LSTM unroll to make seq prediction
            x = target_y[:, i, :] if unroll else mu
            h, c = self.rnn(x, (h, c))  # h (b x hidden_dim)
            mu = self.out(h)  # (b, y_dim)
            mus.append(mu.unsqueeze(1))
            loss += torch.mean((mu - target_y[:, i, :])**2)

        mu = torch.cat(mus, dim=1).detach()
        sigma = torch.zeros_like(mu).detach() # variances are all zero
        return mu, sigma, loss, kl_loss
