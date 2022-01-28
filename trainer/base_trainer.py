import torch
from pathlib import Path
from abc import abstractclassmethod
from functools import partial
from torch.utils.tensorboard.writer import SummaryWriter
import atexit
from utils.logger import write_log


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model,
                 criterion,
                 metric_fns,
                 optimizer,
                 dataset_stats,
                 trainer_cfg,
                 resume_path):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.trainer_cfg = trainer_cfg
        self.num_epochs = trainer_cfg["num_epochs"]

        self.ckpts_dir = Path(trainer_cfg["ckpts_dir"])

        self.early_stop = trainer_cfg["early_stop"]
        self.patience = trainer_cfg["patience"]

        self.metric_fns = metric_fns
        self.write_log = partial(write_log, trainer_cfg["log_file"])

        self.dataset_stats = dataset_stats
        self.start_epoch = 1
        self.save_period = trainer_cfg["save_period"]
        if trainer_cfg["tb_dir"] is not None:
            tb_dir = Path(trainer_cfg["tb_dir"])
            self.train_tb_writer = SummaryWriter(tb_dir / "train")
            self.val_tb_writer = SummaryWriter(tb_dir / "val")
        
        atexit.register(self.cleanup)
        
        if resume_path:
            self._resume_checkpoint(Path(resume_path))

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


    def cleanup(self):
        self.train_tb_writer.close()
        self.val_tb_writer.close()

    def train(self):
        """
        full training logic
        """
        self.write_log("... Training demos are:\n")
        for demo in self.dataloader.get_fnames():
            self.write_log(f"   {demo}\n")
        self.write_log('\n... Training neural network\n')

        best_val_loss = None
        patience_count = 0

        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
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

            # early stopping
            if self.early_stop:
                if epoch == self.start_epoch:
                    best_val_loss = val_log["loss"]
                elif val_log["loss"] <= best_val_loss:
                    best_val_loss = val_log["loss"]
                    patience_count = 0

                    # save ckpt if getting better
                    ckpt_ls = list(self.ckpts_dir.glob("*.pth"))
                    if len(ckpt_ls) > 0:
                        ckpt_ls[0].unlink()
                    self._save_checkpoint(epoch)
                else:
                    self.write_log('... Network is not improving\n')
                    patience_count += 1
                    if patience_count >= self.patience:
                        break
            else:
                if epoch % self.save_period == 0:
                    self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch):
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
            'optimizer': self.optimizer.state_dict(),
            'dataset_stats': self.dataset_stats
        }
        fname = self.ckpts_dir / f'ckpt-epoch{epoch}.pth'
        torch.save(state, fname)
        self.write_log(f'... Saving checkpoint: {fname}\n')

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        # resume_path = str(resume_path)
        resume_path = self._find_ckpt(resume_path)
        self.write_log(f"... Loading Checkpoint: {resume_path}\n")
        ckpt = torch.load(resume_path)
        self.start_epoch = ckpt['epoch'] + 1
        self.model.load_state_dict(ckpt['state_dict'])
        # self.optimizer.load_state_dict(ckpt['optimizer'])

        self.write_log(f"Checkpoint loaded. "
                       f"Resume training from epoch {self.start_epoch}")

    def _find_ckpt(self, ckpt_dir):
        ckpt_dir = Path(ckpt_dir)
        ckpt_pths = [pth for pth in list(ckpt_dir.glob("*.pth"))]
        return ckpt_pths[0]