import time
import argparse
import collections
import numpy as np
from datetime import datetime

from torch import nn
import torch
from data.gp_curves import GPSeqCurvesReader

from models.encoders import DeterministicLSTMEncoder
from models.decoders import DeterministicLSTMDecoder
from models.encoders import LatentLSTMEncoder

from utils.plotting import plot_gp_draws

from utils.logger import Logger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ANP_RNN_Model(nn.Module):
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

    def forward(self, context_x, context_y, target_x, target_y, testing=False):
        # If testing is True we unroll the LSTM and use the previous timestep predictions
        # as an input for a new prediction.

        r = self._deter_encoder(context_x, context_y, target_x if self.use_cross_attention else None)

        if self.latent_path:
            # TODO pass state from latent encoder context to latent encoding of the targets
            z, prior_dist, _     = self._lat_encoder(context_x, context_y) # z (b, latent_dim)
            z_post, post_dist, _ = self._lat_encoder(target_x.unsqueeze(-1), target_y.unsqueeze(-1))
        else:
            z = None
        # If we want to calculate the log_prob for training we will make use of the
        # target_y. At test time the target_y is not available so we return None.
        batch_size, len_target_y, _ = target_y.size()

        dist, mu, sigma = self._decoder(r, target_x, target_y, z, testing=testing)
        #kl = torch.distributions.kl_divergence(post_dist, prior_dist).mean(-1)

        # KW commented out for now
        loss, kl_loss = 0, 0
        # for i in range(len_target_y):
        #     loss += - dist[i].log_prob(target_y[:, i, :]).squeeze(1)
        # if self.latent_path:
        #     kl_loss = torch.sum(torch.distributions.kl.kl_divergence(post_dist, prior_dist), 1) # [100]
        #     loss += kl_loss / batch_size

        # loss = loss.mean()

        # mu = torch.cat(mu, dim=1).detach()
        # sigma = torch.cat(sigma, dim=1).detach()

        loss = - dist.log_prob(target_y[:, -1, :]).squeeze(1)
        if self.latent_path:
            kl_loss = torch.sum(torch.distributions.kl.kl_divergence(post_dist, prior_dist), 1) # [100]
            loss += kl_loss / batch_size

        loss = loss.mean()

        mu = mu.detach()
        sigma = sigma.detach()
        return mu, sigma, loss, kl_loss.mean() if self.latent_path else kl_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Recurrent NP")

    parser.add_argument('--tag', type=str, default='', help='Unique string for TB and plotting.')
    parser.add_argument('--logdir', type=str, default='runs', help='TB log directory.')
    parser.add_argument('--x_attn', default=False, action='store_true',
                        help='Whether to use cross attention between the context and targets.')
    parser.add_argument('--self_attn', default=False, action='store_true',
                        help='Whether to use self attention on the contexts.')
    parser.add_argument('--latent_path', default=False, action='store_true',
                        help='Whether to use the latent path.')
    args = parser.parse_args()

    TRAINING_ITERATIONS = int(2e5)
    PLOT_AFTER = int(1e3)
    num_context = 3
    len_context = 6
    batch_size = 256
    encoder_output_sizes = [64, 64]
    decoder_output_sizes = [64, 64]
    embedding_dim = 64
    latent_dim = 4

    # Define the model
    encoder_input_size = 1 + 1  # x and y pairs are encoder into the context
    decoder_input_size = embedding_dim + 1 + 1  # target is concatenated onto the representation as input into the decoder

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logger = Logger(logdir=args.logdir, run_name=f"{current_time}-{args.tag}")

    model = ANP_RNN_Model(x_dim=1,
                          y_dim=1,
                          encoder_rnn_hidden_size_list=encoder_output_sizes,
                          decoder_rnn_hidden_size_list=decoder_output_sizes,
                          embedding_dim=embedding_dim,
                          latent_dim=latent_dim,
                          context_concat='stack',
                          use_cross_attention=args.x_attn,
                          use_self_attention=args.self_attn,
                          de_cross_attention_type='dot',
                          de_self_attention_type="dot",
                          latent_path=args.latent_path,
                          )

    model = model.to(device)

    # Set up the optimizer and train step
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    timings = collections.deque(maxlen=10)
    losses = []

    for it in range(TRAINING_ITERATIONS):
        start_time = time.time()

        # Train dataset
        dataset_train = GPSeqCurvesReader(batch_size=batch_size,
                                          num_contexts=num_context,
                                          len_context_x=len_context,
                                          len_context_y=len_context,
                                          len_target_x=len_context,
                                          len_target_y=len_context,
                                          testing=False)
        data_train = dataset_train.generate_curves()

        # Test dataset
        dataset_test = GPSeqCurvesReader(batch_size=1,
                                         num_contexts=num_context,
                                         len_context_x=len_context,
                                         len_context_y=len_context,
                                         len_target_x=len_context,
                                         len_target_y=len_context,
                                         testing=False)
        data_test = dataset_test.generate_curves()

        (context_x, context_y), target_x = data_train.query

        context_x = context_x.to(device)
        context_y = context_y.to(device)
        target_x = target_x.to(device)
        target_y = data_train.target_y.to(device)

        pred_y, _, train_loss, kl = model.forward(
            context_x,
            context_y,
            target_x,
            target_y,
        )
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        pred_y_test, _, test_loss, _ = model.forward(
            context_x,
            context_y,
            target_x,
            target_y,
            testing=True,
        )
        train_mse = torch.mean((pred_y - target_y[:, -1:, :])**2)
        test_mse  = torch.mean((pred_y_test - target_y[:, -1:, :])**2)
        

        timings.append(time.time() - start_time)

        logger.log_data(train_loss.item(), kl.item() if args.latent_path else kl,
                        train_mse, test_loss.item(), test_mse, np.mean(timings), it)

        # Plot the predictions in `PLOT_AFTER` intervals
        if it % PLOT_AFTER == 0:
            (context_x, context_y), target_x = data_test.query
            context_x = context_x.to(device)
            context_y = context_y.to(device)
            target_x = target_x.to(device)
            target_y = data_test.target_y.to(device)

            # Get the predicted mean and variance at the target points for the testing set
            pred_y, var, loss, kl = model.forward(
                context_x,
                context_y,
                target_x,
                target_y)
            test_loss = loss.detach().item()
            losses.append(test_loss)

            # Plot the prediction and the context
            plot_gp_draws(
                num_context,
                data_test.left_intervals,
                data_test.len_seqs,
                data_test.full,
                target_x.cpu().numpy(),
                target_y.cpu().numpy(),
                context_x.cpu().numpy(),
                context_y.cpu().numpy(),
                pred_y.cpu().numpy(),
                var.cpu().numpy(),
                tag="{0}_{1}".format(args.tag, str(it)),
            )







