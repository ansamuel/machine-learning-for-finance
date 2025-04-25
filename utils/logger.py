import json
import os
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Logger():
    def __init__(self, logdir, run_name, args):
        self.log_name = logdir + '/' + run_name
        self.tf_writer = None
        self.start_time = time.time()

        if not os.path.exists(self.log_name):
            os.makedirs(self.log_name)

        self.writer = SummaryWriter(self.log_name)

        with open(os.path.join(self.log_name, 'config.txt'), 'w') as file:
            file.write(json.dumps(vars(args)))

    def log_data(self, train_loss, train_loss_unroll, test_loss,
                 train_mse, train_mse_unroll, test_mse,
                 test_kl, update_time, step):
        self.writer.add_scalar(tag="train_loss", scalar_value=train_loss, global_step=step)
        self.writer.add_scalar(tag="train_loss_unroll", scalar_value=train_loss_unroll, global_step=step)
        self.writer.add_scalar(tag="test_loss", scalar_value=test_loss, global_step=step)
        self.writer.add_scalar(tag="test_kl_term", scalar_value=test_kl, global_step=step)
        self.writer.add_scalar(tag="train_mse", scalar_value=train_mse, global_step=step)
        self.writer.add_scalar(tag="train_mse_unroll", scalar_value=train_mse_unroll, global_step=step)
        self.writer.add_scalar(tag="test_mse", scalar_value=test_mse, global_step=step)
        self.writer.add_scalar(tag="update_time", scalar_value=update_time, global_step=step)

