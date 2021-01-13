from abc import abstractclassmethod
from logger import TensorboardWriter


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, optimizer, config):
        self.config = config
        self.logger = config.get_logger('train', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger,
                                        cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractclassmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        Params:
            current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._train_epoch(epoch)

    def _save_checkpoint(self, epoch):
        """
        saving checkpoints
        Params:
            epoch: current epoch number
            log: logging information of the epoch
        """
        raise NotImplementedError

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        Params:
            resume_path: Checkpoint path to be resumed
        """
        raise NotImplementedError
