import numpy as np
import torch
import collections

CNPRegressionDescription = collections.namedtuple(
    "CNPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points"))

CNPSeqRegressionDescription = collections.namedtuple(
    "CNPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points", "left_intervals", "len_seqs", "full"))


class GPCurvesReader(object):
    """Generates curves using a Gaussian Process (GP).

    Supports vector inputs (x) and vector outputs (y). Kernel is
    mean-squared exponential, using the x-value l2 coordinate distance scaled by
    some factor chosen randomly in a range. Outputs are independent gaussian
    processes.
    """

    def __init__(self,
               batch_size,
               max_num_context,
               x_size=1,
               y_size=1,
               l1_scale=0.4,
               sigma_scale=1.0,
               testing=False):
        """Creates a regression dataset of functions sampled from a GP.

        Args:
          batch_size: An integer.
          max_num_context: The max number of observations in the context.
          x_size: Integer >= 1 for length of "x values" vector.
          y_size: Integer >= 1 for length of "y values" vector.
          l1_scale: Float; typical scale for kernel distance function.
          sigma_scale: Float; typical scale for variance.
          testing: Boolean that indicates whether we are testing. If so there are
              more targets for visualization.
        """
        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_size = x_size
        self._y_size = y_size
        self._l1_scale = l1_scale
        self._sigma_scale = sigma_scale
        self._testing = testing

    def _gaussian_kernel(self, xdata, l1, sigma_f, sigma_noise=2e-2):
        """Applies the Gaussian kernel to generate curve data.

        Args:
          xdata: Tensor with shape `[batch_size, num_total_points, x_size]` with
              the values of the x-axis data.
          l1: Tensor with shape `[batch_size, y_size, x_size]`, the scale
              parameter of the Gaussian kernel.
          sigma_f: Float tensor with shape `[batch_size, y_size]`; the magnitude
              of the std.
          sigma_noise: Float, std of the noise that we add for stability.

        Returns:
          The kernel, a float tensor with shape
          `[batch_size, y_size, num_total_points, num_total_points]`.
        """
        num_total_points = xdata.shape[1]

        # Expand and take the difference
        xdata1 = torch.unsqueeze(xdata, axis=1)  # [B, 1, num_total_points, x_size]
        xdata2 = torch.unsqueeze(xdata, axis=2)  # [B, num_total_points, 1, x_size]
        diff = xdata1 - xdata2  # [B, num_total_points, num_total_points, x_size]

        # [B, y_size, num_total_points, num_total_points, x_size]
        norm = torch.square(diff[:, None, :, :, :] / l1[:, :, None, None, :])

        norm = torch.sum(norm, -1)  # [B, data_size, num_total_points, num_total_points]

        # [B, y_size, num_total_points, num_total_points]
        kernel = torch.square(sigma_f)[:, :, None, None] * torch.exp(-0.5 * norm)

        # Add some noise to the diagonal to make the cholesky work.
        kernel += (sigma_noise**2) * torch.eye(num_total_points)

        return kernel

    def generate_curves(self):
        """Builds the op delivering the data.

        Generated functions are `float32` with x values between -2 and 2.

        Returns:
          A `CNPRegressionDescription` namedtuple.
        """
        num_context = torch.randint(
            low=3, high=self._max_num_context, size=(1,), dtype=torch.int32)

        # If we are testing we want to have more targets and have them evenly
        # distributed in order to plot the function.
        if self._testing:
            num_target = 401
            num_total_points = num_target
            x_values = torch.tile(
                torch.unsqueeze(torch.arange(start=-2., end=2., step=1. / 100, dtype=torch.float32), axis=0),
                (self._batch_size, 1))
            x_values = torch.unsqueeze(x_values, axis=-1)
        # During training the number of target points and their x-positions are
        # selected at random
        else:
            num_target = torch.randint(
                low=2, high=self._max_num_context, size=(1,), dtype=torch.int32)
            num_total_points = num_context + num_target
            # (b, c + t)
            # (r1 - r2) * torch.rand(a, b) + r2
            x_values = (-2 - 2) * torch.rand(self._batch_size, num_total_points, self._x_size) + 2

        # Set kernel parameters
        l1 = torch.ones(self._batch_size, self._y_size, self._x_size) * self._l1_scale
        sigma_f = torch.ones(self._batch_size, self._y_size) * self._sigma_scale

        # Pass the x_values through the Gaussian kernel
        # [batch_size, y_size, num_total_points, num_total_points]
        kernel = self._gaussian_kernel(x_values, l1, sigma_f)
        # Calculate Cholesky, using double precision for better stability:
        cholesky = torch.cholesky(kernel.type(torch.DoubleTensor)).type(torch.FloatTensor)

        # Sample a curve
        # [batch_size, y_size, num_total_points, 1]
        y_values = torch.matmul(
            cholesky,
            torch.normal(mean=torch.zeros(self._batch_size, self._y_size, num_total_points, 1),
                         std=1)
        )

        # [batch_size, num_total_points, y_size]
        y_values = torch.squeeze(y_values, 3).permute(0, 2, 1)

        if self._testing:
            # Select the targets
            target_x = x_values
            target_y = y_values

            # Select the observations
            idx = torch.randperm(num_target)
            context_x = x_values[:, idx[:num_context], :]
            context_y = y_values[:, idx[:num_context], :]

        else:
            # Select the targets which will consist of the context points as well as
            # some new target points
            target_x = x_values[:, :num_target + num_context, :]
            target_y = y_values[:, :num_target + num_context, :]

            # Select the observations
            context_x = x_values[:, :num_context, :]
            context_y = y_values[:, :num_context, :]

        query = ((context_x, context_y), target_x)

        return CNPRegressionDescription(
            query=query,
            target_y=target_y,
            num_total_points=target_x.shape[1],
            num_context_points=num_context)


class GPSeqCurvesReader(GPCurvesReader):
    """

    """

    def __init__(self,
                 batch_size,
                 num_contexts,
                 len_context_x,
                 len_context_y,
                 len_target_x,
                 len_target_y,
                 len_target_x_min,
                 x_size=1,
                 y_size=1,
                 l1_scale=0.4,
                 sigma_scale=1.0,
                 testing=False):
        """Creates seq regression dataset of functions sampled from a GP,
        where points are consecutive.

        :param batch_size: int
        :param len_context_x: the seq len in the context x.
        :param len_context_y: the seq len in the context y.
        :param len_target_x: the seq len of the target x
        :param len_target_y: the seq len of the target x
        :param x_size: int >= 1 for the length of the "x vals".
        :param y_size: int >= 1 for the length of the "y vals".
        :param l1_scale: float; scale for the kernel distance.
        :param sigma_scale: flow the scale for the variance.
        :param testing: bool. indicates whether we are testing, if so there are
            more targets for visualization.
        """

        super(GPSeqCurvesReader, self).__init__(batch_size,
                                                len_context_x, # the len of the context x and y as well
                                                x_size,
                                                y_size,
                                                l1_scale,
                                                sigma_scale,
                                                )
        self._len_context_x_max = len_context_x
        self._len_target_x_max = len_target_x
        self._len_target_x_min = len_target_x_min
        self.num_contexts = num_contexts

    def generate_multiple_curves(self):
        """ Builds the op for delivering the seq data with contexts from multiple GP curves.

        Generated functions are `float32` with x values between -4 and 4.

        :return:
            A `CNPSeqRegressionDescription` namedtuple.
        """

        # Randomly draw the length of the context x
        _len_context_x = torch.randint(low=10, high=self._len_context_x_max, size=(1,), dtype=torch.int32)

        # need to stack x and y for embedding into the RNN, so these need to be the same length
        _len_context_y = _len_context_x

        # Randomly draw the length of the target x and ys
        # target_x and target_y need to be the same as the KL term in the latent encoder requires
        # passing contexts and targets through the encoder and the stacking operation requires
        # lengths of the x and y's to be the same.
        _len_target_x = torch.randint(low=self._len_target_x_min, high=self._len_target_x_max, size=(1,), dtype=torch.int32)
        _len_target_y = _len_target_x

        # Creating the x range
        # 2 for split between c and t
        # 2 for good measure
        # 1/10 is each step in x
        r = (_len_target_x.item() * 2 * 2 * 2) // 10  # int

        x_values = torch.tile(
            torch.unsqueeze(torch.arange(start=-r, end=r, step=1. / 10, dtype=torch.float32), axis=0),
            (self._batch_size, 1)
        )
        _, seq_len = x_values.size()
        x_values = torch.unsqueeze(x_values, axis=-1)

        # store the contexts and full GP draws
        context_len = _len_context_x + _len_context_y
        target_len = _len_target_x + _len_target_y
        context_x = torch.ones(self._batch_size, _len_context_x, self._y_size, self.num_contexts)
        context_y = torch.ones(self._batch_size, _len_context_y, self._y_size, self.num_contexts)
        y_values_full = torch.ones(self._batch_size, seq_len, self._y_size, self.num_contexts + 1)
        left_intervals = []

        # Set kernel parameters
        l1 = torch.ones(self._batch_size, self._y_size, self._x_size) * self._l1_scale
        sigma_f = torch.ones(self._batch_size, self._y_size) * self._sigma_scale

        # Pass the x_values through the Gaussian kernel
        # [batch_size, y_size, num_total_points, num_total_points]
        kernel = self._gaussian_kernel(x_values, l1, sigma_f)

        # Calculate Cholesky, using double precision for better stability:
        cholesky = torch.linalg.cholesky(kernel.type(torch.DoubleTensor)).type(torch.FloatTensor)

        # Sample a curve
        # we need num_contexts curves and one more for the target
        # [batch_size, y_size, num_total_points, num_contexts + 1]
        y_values = torch.matmul(
            cholesky,
            torch.normal(mean=torch.zeros(self._batch_size, self._y_size, seq_len, self.num_contexts + 1), std=1),
        )

        # [batch_size, num_total_points, y_size, num_contexts]
        #y_values = torch.squeeze(y_values, 3).permute(0, 2, 1)
        y_values_full = y_values.permute(0, 2, 1, 3)

        # slice y_values to construct the contexts and targets
        start = torch.randint(low=0, high=(seq_len - context_len.item() - 1), size=(1,), dtype=torch.int32)
        context_x = y_values_full[:, start:start + _len_context_x, :, :-1]
        context_y = y_values_full[:, start + _len_context_x:start + context_len, :, :-1]
        left_intervals += [start] * self.num_contexts

        start = torch.randint(low=0, high=(seq_len - target_len.item() - 1), size=(1,), dtype=torch.int32)
        target_x = y_values_full[:, start:(start + _len_target_x), :, -1]
        target_y = y_values_full[:, (start + _len_target_x):(start + target_len), :, -1]
        left_intervals.append(start)

        # left_intervals = []
        # for i in range(self.num_contexts + 1):
        #     # Set kernel parameters
        #     l1 = torch.ones(self._batch_size, self._y_size, self._x_size) * self._l1_scale
        #     sigma_f = torch.ones(self._batch_size, self._y_size) * self._sigma_scale
        #
        #     # Pass the x_values through the Gaussian kernel
        #     # [batch_size, y_size, num_total_points, num_total_points]
        #     kernel = self._gaussian_kernel(x_values, l1, sigma_f)
        #
        #     # Calculate Cholesky, using double precision for better stability:
        #     cholesky = torch.linalg.cholesky(kernel.type(torch.DoubleTensor)).type(torch.FloatTensor)
        #
        #     # Sample a curve
        #     # [batch_size, y_size, num_total_points, 1]
        #     y_values = torch.matmul(
        #         cholesky,
        #         torch.normal(mean=torch.zeros(self._batch_size, self._y_size, seq_len, 1), std=1),
        #     )
        #
        #     # [batch_size, num_total_points, y_size]
        #     y_values = torch.squeeze(y_values, 3).permute(0, 2, 1)
        #
        #     # slice y_values to construct the contexts and targets
        #     if i != self.num_contexts:
        #         start = torch.randint(low=0, high=(seq_len - context_len.item() - 1), size=(1,), dtype=torch.int32)
        #         context_x[:, :, :, i] = y_values[:, start:start + _len_context_x, :]
        #         context_y[:, :, :, i] = y_values[:, start + _len_context_x:start + context_len, :]
        #     else:
        #         start = torch.randint(low=0, high=(seq_len - target_len.item() - 1), size=(1,), dtype=torch.int32)
        #         target_x = y_values[:, start:(start + _len_target_x), :]
        #         target_y = y_values[:, (start + _len_target_x):(start + target_len), :]
        #     left_intervals.append(start)
        #     # for plotting how the targets and contexts are generated from the GP draws we store the entire GP draw
        #     y_values_full[:, :, :, i] = y_values

        query = ((context_x, context_y), target_x)

        return CNPSeqRegressionDescription(
            query=query,
            target_y=target_y,
            num_total_points=target_x.shape[1],
            num_context_points=self.num_contexts,
            left_intervals=left_intervals,
            len_seqs=(_len_context_x,
                      _len_context_y,
                      _len_target_x,
                      _len_target_y),
            full=y_values_full,
        )


    def generate_curves(self):
        """ Builds the op for delivering the seq data.

        Generated functions are `float32` with x values between -4 and 4.

        :return:
            A `CNPSeqRegressionDescription` namedtuple.
        """

        # Randomly draw the length of the context x
        _len_context_x = torch.randint(low=10, high=self._len_context_x_max, size=(1,), dtype=torch.int32)

        # need to stack x and y for embedding into the RNN, so these need to be the same length
        _len_context_y = _len_context_x

        # Randomly draw the length of the target x and ys
        # target_x and target_y need to be the same as the KL term in the latent encoder requires
        # passing contexts and targets through the encoder and the stacking operation requires
        # lengths of the x and y's to be the same.
        _len_target_x = torch.randint(low=self._len_target_x_min, high=self._len_target_x_max, size=(1,), dtype=torch.int32)
        _len_target_y = _len_target_x

        # Creating the x range
        # 2 for split between c and t
        # 2 for good measure
        # 1/10 is each step in x
        r = (_len_context_y.item() * 2 * 2 * 2) // 10  # int

        x_values = torch.tile(
            torch.unsqueeze(torch.arange(start=-r, end=r, step=1. / 10, dtype=torch.float32), axis=0),
            (self._batch_size, 1)
        )
        _, seq_len = x_values.size()

        l_context = seq_len // 2

        x_values = torch.unsqueeze(x_values, axis=-1)

        # Do create n contexts from the GP draw, let's randomly sample 3 different left limits to the intervals
        # for each context within 1/n of each function draw
        left_intervals = []
        for i in range(self.num_contexts):
            left_intervals.append(torch.randint(low=i * (l_context // self.num_contexts),
                                                high=(i + 1) * (l_context // self.num_contexts) - _len_context_x.item() - 1,
                                                size=(1,),
                                                dtype=torch.int32))

        # Set kernel parameters
        l1 = torch.ones(self._batch_size, self._y_size, self._x_size) * self._l1_scale
        sigma_f = torch.ones(self._batch_size, self._y_size) * self._sigma_scale

        # Pass the x_values through the Gaussian kernel
        # [batch_size, y_size, num_total_points, num_total_points]
        kernel = self._gaussian_kernel(x_values, l1, sigma_f)

        # Calculate Cholesky, using double precision for better stability:
        cholesky = torch.linalg.cholesky(kernel.type(torch.DoubleTensor)).type(torch.FloatTensor)

        # Sample a curve
        # [batch_size, y_size, num_total_points, 1]
        y_values = torch.matmul(
            cholesky,
            torch.normal(mean=torch.zeros(self._batch_size, self._y_size, seq_len, 1), std=1),
        )

        # [batch_size, num_total_points, y_size]
        y_values = torch.squeeze(y_values, 3).permute(0, 2, 1)

        # slice y_values to construct the contexts and targets
        context_len = _len_context_x + _len_context_y
        target_len = _len_target_x + _len_target_y
        context_x = torch.ones(self._batch_size, _len_context_x, self._y_size, self.num_contexts)
        context_y = torch.ones(self._batch_size, _len_context_y, self._y_size, self.num_contexts)
        for i in range(self.num_contexts):
            l = left_intervals[i]
            context_x[:, :, :, i] = y_values[:, l:l + _len_context_x, :]
            context_y[:, :, :, i] = y_values[:, l + _len_context_x:l + context_len, :]

        target_x = y_values[:, l_context:l_context + _len_target_x, :]
        target_y = y_values[:, l_context + _len_target_x:l_context + target_len, :]

        query = ((context_x, context_y), target_x)

        return CNPSeqRegressionDescription(
            query=query,
            target_y=target_y,
            num_total_points=target_x.shape[1],
            num_context_points=self.num_contexts,
            left_intervals=left_intervals + [l_context],
            len_seqs=(_len_context_x,
                      _len_context_y,
                      _len_target_x,
                      _len_target_y),
            full=y_values,
        )



if __name__ == "__main__":
    TRAINING_ITERATIONS = int(2e5)
    MAX_CONTEXT_POINTS = 10
    PLOT_AFTER = int(2e4)
    NUM_CONTEXT = 3
    CONTEXT_LEN = 10
    TARGET_LEN_MAX = 20
    TARGET_LEN_MIN = 10
    # Train dataset
    dataset_train = GPSeqCurvesReader(
        batch_size=64,
        num_contexts=NUM_CONTEXT,
        len_context_x=TARGET_LEN_MAX,
        len_context_y=TARGET_LEN_MAX,
        len_target_x=TARGET_LEN_MAX,
        len_target_y=TARGET_LEN_MAX,
        len_target_x_min=TARGET_LEN_MIN)
    data_train = dataset_train.generate_multiple_curves()
    (context_x, context_y), target_x = data_train.query
