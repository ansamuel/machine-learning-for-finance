from torch import nn
import torch

from models.encoders import DeterministicLSTMEncoder
from models.decoders import DeterministicLSTMDecoder
from models.encoders import LatentLSTMEncoder

class SeqNPModel(nn.Module):
    """
    (Attentive) Neural Process model
    https://github.com/VersElectronics/Neural-Processes/blob/master/neural_process_models/anp_rnn.py
    """

    def __init__(self,
                 x_dim,
                 y_dim,
                 encoder_rnn_hidden_size_list,
                 decoder_rnn_hidden_size_list,
                 embedding_dim,
                 latent_dim,
                 context_concat='stack',
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
        super(SeqNPModel, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.encoder_rnn_hidden_size_list = encoder_rnn_hidden_size_list
        self.decoder_rnn_hidden_size_list = decoder_rnn_hidden_size_list
        self.embedding_dim = embedding_dim # same as rnn hidden dim
        self.num_rnn_layers = 1
        self.use_cross_attention = use_cross_attention
        self.latent_dim = latent_dim
        self.latent_path = latent_path

        self._deter_encoder = DeterministicLSTMEncoder(
            x_dim=self.x_dim,
            y_dim=self.y_dim,
            hidden_dim_list=self.encoder_rnn_hidden_size_list,
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
            embedding_dim=self.embedding_dim + self.latent_dim if self.latent_path else self.embedding_dim,
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

    def forward(self, context_x, context_y, target_x, target_y, unroll=False):
        # If testing is True we unroll the LSTM and use the previous timestep predictions
        # as an input for a new prediction.

        r = self._deter_encoder(context_x, context_y, target_x if self.use_cross_attention else None)

        if self.latent_path:
            # TODO pass state from latent encoder context to latent encoding of the targets
            z, prior_dist = self._lat_encoder(context_x, context_y) # z (b, latent_dim)
            z_post, post_dist = self._lat_encoder(target_x.unsqueeze(-1), target_y.unsqueeze(-1))
        else:
            z = None
        # If we want to calculate the log_prob for training we will make use of the
        # target_y. At test time the target_y is not available so we return None.
        batch_size, len_target_y, _ = target_y.size()

        dist, mu, sigma = self._decoder(r, target_x, target_y, z, unroll=unroll)
        loss, kl_loss = 0, 0
        for i in range(len_target_y):
            loss += - dist[i].log_prob(target_y[:, i, :]).squeeze(1)
        if self.latent_path:
            kl_loss = torch.sum(torch.distributions.kl.kl_divergence(post_dist, prior_dist), 1) # [100]
            loss += kl_loss / batch_size

        loss = loss.mean()

        mu = torch.cat(mu, dim=1).detach()
        sigma = torch.cat(sigma, dim=1).detach()
        return mu, sigma, loss, kl_loss.mean() if self.latent_path else kl_loss