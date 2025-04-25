import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
import matplotlib.pyplot as plt
import collections
from data.gp_curves import GPCurvesReader, CNPRegressionDescription

# The CNP takes as input a `CNPRegressionDescription` namedtuple with fields:
#   `query`: a tuple containing ((context_x, context_y), target_x)
#   `target_y`: a tesor containing the ground truth for the targets to be
#     predicted
#   `num_total_points`: A vector containing a scalar that describes the total
#     number of datapoints used (context + target)
#   `num_context_points`: A vector containing a scalar that describes the number
#     of datapoints used as context
# The GPCurvesReader returns the newly sampled data in this format at each
# iteration

class DeterministicEncoder(nn.Module):
    """The Encoder."""

    def __init__(self, input_size):
        """CNP encoder.

        Args:
          output_sizes: An iterable containing the output sizes of the encoding MLP.
        """
        super(type(self), self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

    def forward(self, context_x, context_y, num_context_points):
        """Encodes the inputs into one representation.

        Args:
          context_x: Tensor of size bs x observations x m_ch. For this 1D regression
              task this corresponds to the x-values.
          context_y: Tensor of size bs x observations x d_ch. For this 1D regression
              task this corresponds to the y-values.
          num_context_points: A tensor containing a single scalar that indicates the
              number of context_points provided in this iteration.

        Returns:
          representation: The encoded representation averaged over all context
              points.
        """

        # Concatenate x and y along the filter axes
        encoder_input = torch.cat((context_x, context_y), axis=-1)
        # Get the shapes of the input and reshape to parallelise across observations
        batch_size, _, filter_size = encoder_input.shape
        hidden = torch.reshape(encoder_input, (batch_size * num_context_points, -1))
        hidden = torch.reshape(hidden, (-1, filter_size))
        hidden = self.encoder(hidden)

        # Bring back into original shape
        hidden = torch.reshape(hidden, (batch_size, num_context_points, 128))

        # Aggregator: take the mean over all points
        representation = torch.mean(hidden, axis=1)

        return representation

class DeterministicDecoder(nn.Module):
    """The Decoder."""

    def __init__(self, input_size):
        """CNP decoder.

        Args:
          output_sizes: An iterable containing the output sizes of the decoder MLP
              as defined in `basic.Linear`.
        """
        super(type(self), self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, representation, target_x, num_total_points):
        """Decodes the individual targets.

        Args:
          representation: The encoded representation of the context
          target_x: The x locations for the target query
          num_total_points: The number of target points.

        Returns:
          dist: A multivariate Gaussian over the target points.
          mu: The mean of the multivariate Gaussian.
          sigma: The standard deviation of the multivariate Gaussian.
        """

        # Concatenate the representation and the target_x
        representation = torch.tile(
            torch.unsqueeze(representation, axis=1), (1, num_total_points, 1))
        input = torch.cat((representation, target_x), axis=-1)

        # Get the shapes of the input and reshape to parallelise across observations
        batch_size, _, filter_size = input.shape
        hidden = torch.reshape(torch.reshape(input, (batch_size * num_total_points, -1)), (-1, filter_size))
        # Last layer without a ReLu
        hidden = self.decoder(hidden)

        # Bring back into original shape
        hidden = torch.reshape(hidden, (batch_size, num_total_points, -1))

        # Get the mean an the variance
        mu, log_sigma = torch.unsqueeze(hidden[:,:,0], 2), torch.unsqueeze(hidden[:,:,1], 2)

        # Bound the variance
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)

        # Get the distribution
        dist = D.Independent(D.Normal(loc=mu, scale=sigma), 1)

        return dist, mu, sigma

class DeterministicModel(nn.Module):
    """The CNP model."""

    def __init__(self, encoder_input_size, decoder_input_size):
        """Initialises the model.

        Args:
          encoder_output_sizes: An iterable containing the sizes of hidden layers of
              the encoder. The last one is the size of the representation r.
          decoder_output_sizes: An iterable containing the sizes of hidden layers of
              the decoder. The last element should correspond to the dimension of
              the y * 2 (it encodes both mean and variance concatenated)
        """
        super(type(self), self).__init__()
        self.encoder = DeterministicEncoder(encoder_input_size)
        self.decoder = DeterministicDecoder(decoder_input_size)

    def forward(self, query, num_total_points, num_contexts, target_y=None):
        """Returns the predicted mean and variance at the target points.

        Args:
          query: Array containing ((context_x, context_y), target_x) where:
              context_x: Array of shape batch_size x num_context x 1 contains the
                  x values of the context points.
              context_y: Array of shape batch_size x num_context x 1 contains the
                  y values of the context points.
              target_x: Array of shape batch_size x num_target x 1 contains the
                  x values of the target points.
          target_y: The ground truth y values of the target y. An array of
              shape batchsize x num_targets x 1.
          num_total_points: Number of target points.

        Returns:
          log_p: The log_probability of the target_y given the predicted
          distribution.
          mu: The mean of the predicted distribution.
          sigma: The variance of the predicted distribution.
        """

        (context_x, context_y), target_x = query

        # Pass query through the encoder and the decoder
        representation = self.encoder(context_x, context_y, num_contexts)
        dist, mu, sigma = self.decoder(representation, target_x, num_total_points)

        # If we want to calculate the log_prob for training we will make use of the
        # target_y. At test time the target_y is not available so we return None
        if target_y is not None:
            log_p = dist.log_prob(target_y)
        else:
            log_p = None

        return log_p, mu, sigma

if __name__ == "__main__":
    TRAINING_ITERATIONS = int(2e5)
    MAX_CONTEXT_POINTS = 10
    PLOT_AFTER = int(2e4)

    # Sizes of the layers of the MLPs for the encoder and decoder
    # The final output layer of the decoder outputs two values, one for the mean and
    # one for the variance of the prediction at the target location
    encoder_output_sizes = [128, 128, 128, 128]
    decoder_output_sizes = [128, 128, 2]

    # Define the model
    encoder_input_size = 1 + 1  # x and y pairs are encoder into the context
    decoder_input_size = 128 + 1  # target is concatenated onto the representation as input into the decoder
    model = DeterministicModel(encoder_input_size, decoder_input_size)

    # Set up the optimizer and train step
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for it in range(TRAINING_ITERATIONS):

        # Train dataset
        dataset_train = GPCurvesReader(
            batch_size=64, max_num_context=MAX_CONTEXT_POINTS)
        data_train = dataset_train.generate_curves()

        # Test dataset
        dataset_test = GPCurvesReader(
            batch_size=1, max_num_context=MAX_CONTEXT_POINTS, testing=True)
        data_test = dataset_test.generate_curves()

        log_prob, pred_y, var = model.forward(
            data_train.query, data_train.num_total_points,
            data_train.num_context_points, data_train.target_y
        )
        optimizer.zero_grad()
        loss = -torch.mean(log_prob)
        loss.backward()
        optimizer.step()

        # Plot the predictions in `PLOT_AFTER` intervals
        if it % PLOT_AFTER == 0:
            (context_x, context_y), target_x = data_test.query
            # Get the predicted mean and variance at the target points for the testing set
            log_prob, pred_y, var = model.forward(
                data_test.query, data_test.num_total_points,
                data_test.num_context_points, data_test.target_y)
            test_loss = -torch.mean(log_prob).detach().item()
            print('Iteration: {}, loss: {}'.format(it, test_loss))