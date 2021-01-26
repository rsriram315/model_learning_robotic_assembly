import torch
import numpy as np
from base import BaseTrainer
from utils import MetricTracker, ensure_dir


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_fns, optimizer,
                 num_epochs, ckpts_dir, log_file, save_period,
                 device, dataloader,
                 valid_dataloader=None,
                 lr_scheduler=None):

        super().__init__(model, criterion, metric_fns, optimizer,
                         num_epochs, ckpts_dir, log_file, save_period)

        for f in [ckpts_dir, log_file]:
            ensure_dir(f)

        self.device = device
        self.dataloader = dataloader

        self.valid_dataloader = valid_dataloader
        self.do_validation = self.valid_dataloader is not None

        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(self.dataloader.batch_size))

        self.train_metrics = \
            MetricTracker('loss', *[m.__name__ for m in self.metric_fns])
        self.valid_metrics = \
            MetricTracker('loss', *[m.__name__ for m in self.metric_fns])

    def _train_epoch(self, epoch):
        """
        training logic for an epoch

        Args:
            epoch: integer, current training epoch.
        Returns:
            a log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (state_action, target) in enumerate(self.dataloader):
            state_action, target = (state_action.to(self.device),
                                    target.to(self.device))

            self.optimizer.zero_grad()  # intialize the gradient to zero first
            output = self.model(state_action)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item())

            for met in self.metric_fns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.write_log(f'Train Epoch: {epoch} '
                               f'{self._progress(batch_idx)} '
                               f'Loss: {loss.item():.6f}')

            if batch_idx == len(self.dataloader):
                break

        log = self.train_metrics.result()

        return log

    def _valid_epoch(self):
        """
        validate after training an epoch

        Params:
            epoch: integer, current training epoch.
        Return:
            A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            for _, (data, target) in enumerate(self.valid_dataloader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_fns:
                    self.valid_metrics.update(met.__name__,
                                              met(output, target))
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        current = batch_idx * self.dataloader.batch_size
        total = self.dataloader.n_samples

        return f'[{current}/{total} ({(100.0 * current / total):.0f}%)]'
