import matplotlib.pyplot as plt

def plot_gp_draws(num_context, left_interval, len_seqs, full, target_x,
                  target_y, context_x, context_y, pred_y, var, tag):
    """Plots the predicted mean and variance and the context points.

    Args:
    num_context: Int, the number of conext time-series.
    left_interval: List, the left most point of the interval of the context
    len_seqs:
    full: the full GP draw for plotting (batchsize x L x 1)
    target_x: An array of shape batchsize x number_targets x 1 that contains the
        x values of the target points.
    target_y: An array of shape batchsize x number_targets x 1 that contains the
        y values of the target points.
    context_x: An array of shape batchsize x number_context x 1 x num_context that contains
        the x values of the context points.
    context_y: An array of shape batchsize x number_context x 1 x num_context that contains
        the y values of the context points.
    pred_y: An array of shape batchsize x number_targets x 1  that contains the
        predicted means of the y values at the target points in target_x.
    pred_y: An array of shape batchsize x number_targets x 1  that contains the
        predicted variance of the y values at the target points in target_x.
    tag: str unique identifier.
    """
    # Get lengths of context and targets, x and y
    lens = []
    for l in len_seqs:
        if len(lens) == 0:
            lens.append(l)
        else:
            lens.append(l + lens[-1])

    # Context
    context_colours = ['royalblue', 'forestgreen', 'darkorange']
    for i in range(num_context):
        plt.plot(range(left_interval[i], left_interval[i] + len_seqs[0]),
                 context_x[0, :, 0, i], color=context_colours[i], markersize=2, label="context_x {}".format(i + 1))
        plt.plot(range(left_interval[i] + len_seqs[0], left_interval[i] + len_seqs[0] + len_seqs[1]),
                 context_y[0, :, 0, i], linestyle=':', color=context_colours[i], markersize=2,
                 label="context_y {}".format(i + 1))
    # Target
    plt.plot(range(left_interval[-1], left_interval[-1] + len_seqs[2]),
             target_x[0, :, 0], color='purple', linewidth=2, label="target_x")
    plt.plot(range(left_interval[-1] + len_seqs[2], left_interval[-1] + len_seqs[2] + len_seqs[3]),
             target_y[0, :, 0], linestyle=':', color='lightpink', linewidth=2, label="target_y")
    # Prediction
    # +1 to the range on the right interval since there are len+1 predictions made by the decoder
    plt.plot(range(left_interval[-1] + len_seqs[2],left_interval[-1] + len_seqs[2]+len(pred_y[0, :, 0])),#, left_interval[-1] + len_seqs[2] + len_seqs[3] + 1),
             pred_y[0, :, 0], 'blue', label="pred")

    plt.fill_between(
        range(left_interval[-1] + len_seqs[2], left_interval[-1] + len_seqs[2] + len_seqs[3] + 1),
        pred_y[0, :, 0] - var[0, :, 0],
        pred_y[0, :, 0] + var[0, :, 0],
        alpha=0.2,
        color='blue',
        interpolate=True)

    # Full
    plt.plot(range(full.shape[1]), full[0, :, 0], linestyle=':', color='grey')

    plt.legend()

    plt.grid(False)
    # ax.set_axis_bgcolor('white')
    plt.savefig('test_gp_draws_{}.png'.format(tag))
