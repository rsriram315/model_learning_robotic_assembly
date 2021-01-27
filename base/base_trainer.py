import torch
from pathlib import Path
from abc import abstractclassmethod
from functools import partial
from torch.utils.tensorboard.writer import SummaryWriter
from logger import write_log


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_fns, optimizer,
                 num_epochs, ckpts_dir, save_period, log_file, tb_dir):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs

        self.ckpts_dir = Path(ckpts_dir)
        self.metric_fns = metric_fns
        self.write_log = partial(write_log, log_file)

        self.save_period = save_period
        if tb_dir is not None:
            tb_dir = Path(tb_dir)
            self.train_tb_writer = SummaryWriter(tb_dir / "train")
            self.val_tb_writer = SummaryWriter(tb_dir / "val")

    @abstractclassmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        Params:
            current epoch number
        """
        raise NotImplementedError

    @abstractclassmethod
    def _valid_epoch(self):
        """
        Validation logic for an epoch

        Params:
            current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        full training logic
        """
        self.write_log('... Training neural network\n')
        for epoch in range(1, self.num_epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            self.write_log(f'Train Epoch: {epoch} '
                           f'[{self.dataloader.n_samples}/'
                           f'{self.dataloader.n_samples} (100%)] '
                           f'Loss: {log["loss"]:.6f}')

            if self.do_validation:
                val_log = self._valid_epoch()
                self.write_log(f'Validation loss: {val_log["loss"]}\n')
                self.val_tb_writer.add_scalar("loss", val_log["loss"],
                                              epoch * len(self.dataloader))

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint
                          to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        fname = self.ckpts_dir / f'ckpt-epoch{epoch}.pth'
        torch.save(state, fname)
        self.write_log(f'... Saving checkpoint: {fname}\n')

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        ckpt = torch.load(resume_path)
        self.model.load_state_dict(ckpt['state_dict'])

        self.optimizer.load_state_dict(ckpt['optimizer'])

        self.write_log(f"Checkpoint loaded. "
                       f"Resume training from epoch {self.start_epoch}")
