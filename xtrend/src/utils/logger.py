import logging
import os
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Logger():
    def __init__(self, logdir, run_name):
        self.log_name = logdir + '/' + run_name
        self.tf_writer = None
        self.start_time = time.time()

        if not os.path.exists(self.log_name):
            os.makedirs(self.log_name)

        self.writer = SummaryWriter(self.log_name)

    def log_data(self, train_loss, kl, train_mse, test_loss, test_mse, update_time, step):
        self.writer.add_scalar(tag="train_loss", scalar_value=train_loss, global_step=step)
        self.writer.add_scalar(tag="test_loss", scalar_value=test_loss, global_step=step)
        self.writer.add_scalar(tag="kl_term", scalar_value=kl, global_step=step)
        self.writer.add_scalar(tag="train_mse", scalar_value=train_mse, global_step=step)
        self.writer.add_scalar(tag="test_mse", scalar_value=test_mse, global_step=step)
        self.writer.add_scalar(tag="update_time", scalar_value=update_time, global_step=step)

