import time
from datetime import datetime
import argparse
import collections
import numpy as np
import torch

from seq_np import SeqNPModel
from lstm_baseline import LSTMBaseline

from data.gp_curves import GPSeqCurvesReader
from utils.plotting import plot_gp_draws, plot_multiple_gp_draws
from utils.logger import Logger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Recurrent NP")

    # Experiment params
    parser.add_argument('--tag', type=str, default='', help='Unique string for TB and plotting.')
    parser.add_argument('--seed', type=int, default=100, help='Sets the random seed.')
    parser.add_argument('--logdir', type=str, default='runs', help='TB log directory.')
    parser.add_argument('--lstm_baseline', default=False, action='store_true',
                        help='Whether to train an LSTM baseline model.')
    parser.add_argument('--unroll', default=True, action='store_true',
                        help='Whether to test by unrolling to make predictions rather than using teacher forcing.')
    parser.add_argument('--max_len_context', type=int, default=20,
                        help='The max len of the context for the GP draws dataset.')
    parser.add_argument('--max_len_target', type=int, default=20,
                        help='The max len of the target for the GP draws dataset.')
    parser.add_argument('--min_len_target', type=int, default=10,
                        help='The min len of the target for the GP draws dataset.')
    parser.add_argument('--num_contexts', type=int, default=3,
                        help='The number of GP function draws in the context set.')
    parser.add_argument('--num_runs', type=int, default=1,
                        help='The number runs to perform.')
    parser.add_argument('--data_gen_many', default=False, action='store_true',
                        help='Whether to draw contexts from multiple GP draws.')

    # Seq NP params
    parser.add_argument('--x_attn', default=False, action='store_true',
                        help='Whether to use cross attention between the context and targets.')
    parser.add_argument('--self_attn', default=False, action='store_true',
                        help='Whether to use self attention on the contexts.')
    parser.add_argument('--latent_path', default=False, action='store_true',
                        help='Whether to use the latent path.')

    args = parser.parse_args()

    print("Device: {0}".format(device))

    # Params
    TRAINING_ITERATIONS = int(5e4)
    num_context = args.num_contexts
    len_context = args.max_len_context
    len_target = args.max_len_target
    batch_size = 256
    encoder_output_sizes = [64, 64]
    decoder_output_sizes = [64, 64]
    embedding_dim = 64
    latent_dim = 4

    # Define the model
    encoder_input_size = 1 + 1  # x and y pairs are encoder into the context
    decoder_input_size = embedding_dim + 1 + 1  # target is concatenated onto the representation as input into the decoder

    for i in range(args.num_runs):

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        logger = Logger(logdir=args.logdir, run_name=f"{current_time}-{args.tag}-s{str(args.seed + i)}", args=args)

        # Set seeds
        torch.manual_seed(args.seed + i)
        np.random.seed(args.seed + i)

        if args.lstm_baseline:
            model = LSTMBaseline(
                x_dim=1,
                y_dim=1,
                hidden_dim_list=decoder_output_sizes,
            )
        else:
            model = SeqNPModel(
                x_dim=1,
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

        # Train dataset
        dataset_train = GPSeqCurvesReader(batch_size=batch_size,
                                          num_contexts=num_context,
                                          len_context_x=len_context,
                                          len_context_y=len_context,
                                          len_target_x=len_target,
                                          len_target_y=len_target,
                                          len_target_x_min=args.min_len_target,
                                          testing=False)

        # Test dataset
        dataset_test = GPSeqCurvesReader(batch_size=1,
                                         num_contexts=num_context,
                                         len_context_x=len_context,
                                         len_context_y=len_context,
                                         len_target_x=len_target,
                                         len_target_y=len_target,
                                         len_target_x_min=args.min_len_target,
                                         testing=False)

        for it in range(TRAINING_ITERATIONS):
            start_time = time.time()

            data_train = dataset_train.generate_multiple_curves() if args.data_gen_many else dataset_train.generate_curves()

            data_test = dataset_test.generate_multiple_curves() if args.data_gen_many else dataset_test.generate_curves()

            (context_x, context_y), target_x = data_train.query

            context_x = context_x.to(device)
            context_y = context_y.to(device)
            target_x = target_x.to(device)
            target_y = data_train.target_y.to(device)

            # Training with teacher forcing
            pred_y, _, train_loss, _ = model.forward(
                context_x,
                context_y,
                target_x,
                target_y,
                unroll=False,
            )

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Test with composite LSTM unrolling on the training set
            with torch.no_grad():
                pred_y_unroll, _, train_loss_unroll, _ = model.forward(
                    context_x,
                    context_y,
                    target_x,
                    target_y,
                    unroll=True,
                )

            train_mse = torch.mean((pred_y[:, :-1, :].detach() - target_y)**2)
            train_mse_unroll = torch.mean((pred_y_unroll[:, :-1, :].detach() - target_y)**2)

            (context_x, context_y), target_x = data_test.query
            context_x = context_x.to(device)
            context_y = context_y.to(device)
            target_x = target_x.to(device)
            target_y = data_test.target_y.to(device)

            # Get the predicted mean and variance at the target points for the testing set
            with torch.no_grad():
                pred_y_test, var, test_loss, test_kl = model.forward(
                    context_x,
                    context_y,
                    target_x,
                    target_y,
                    unroll=args.unroll,
                )

                test_loss = test_loss.detach()

            test_mse = torch.mean((pred_y_test[:, :-1, :].detach() - target_y)**2)

            timings.append(time.time() - start_time)

            logger.log_data(train_loss.item(), train_loss_unroll.item(),
                            test_loss.item(), train_mse, train_mse_unroll,
                            test_mse, test_kl.item() if args.latent_path else 0,
                            np.mean(timings), it)

            if it % 1000 == 0:
                print("Round {0} / {1} / Iter {2} / {3}, Test loss {4:.2f} / Test MSE {5:.2f}".format(
                    i, args.num_runs, it, TRAINING_ITERATIONS, test_loss.item(), test_mse
                    )
                )

        plt_fnc = plot_multiple_gp_draws if args.data_gen_many else plot_gp_draws
        plt_fnc(
            num_context,
            data_test.left_intervals,
            data_test.len_seqs,
            data_test.full,
            target_x.cpu().numpy(),
            target_y.cpu().numpy(),
            context_x.cpu().numpy(),
            context_y.cpu().numpy(),
            pred_y_test.cpu().numpy(),
            var.cpu().numpy(),
            tag="{0}_s{1}_{2}".format(args.tag, str(args.seed + i), str(it)),
        )
