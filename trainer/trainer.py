import torch
from base import BaseTrainer
from utils import inf_loop


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, optimizer, config, device,
                 data_loader):
        super().__init__(model, criterion, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        # iteration-based training
        self.data_loader = inf_loop(data_loader)

    def _train_epoch(self, epoch):
        """
        training logic for an epoch

        Params:
            epoch: integer, current training epoch.
        Returns:
            a log that contains average loss and metric in this epoch.
        """
        self.model.train()
        # TODO: reset metric
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()  # intialize the gradient to zero first
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
        # TODO: logging info

    def _valid_epoch(self, epoch):
        """
        validate after training an epoch

        Params:
            epoch: integer, current training epoch.
        Return: 
            A log that contains information about validation
        """
        self.model.eval()
        # TODO: reset metrics
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                # TODO: logging info

    def _progress(self, batch_idx):
        current = batch_idx * self.data_loader.batch_size
        total = self.data_loader.n_samples
        return f'[{current}/{total} ((100.0 * current / total):.0f)]'
