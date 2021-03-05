import os
from trainer import Trainer


class EnsembleTrainer:
    """
    Ensemble Trainer class
    """
    def __init__(self,
                 dataloader,
                 dataset_stats,
                 trainer_cfg,
                 optim_cfg,
                 model_cfg,
                 valid_dataloader=None,
                 lr_scheduler=None):

        self.dataloader = dataloader
        self.dataset_stats = dataset_stats

        self.trainer_cfg = trainer_cfg
        self.optim_cfg = optim_cfg
        self.model_cfg = model_cfg

        self.valid_dataloader = valid_dataloader
        self.lr_scheduler = lr_scheduler

    def train(self):
        num_ensemble = self.trainer_cfg["num_ensemble"]
        dir_prefix = self.trainer_cfg["ckpts_dir"]

        for n in range(num_ensemble):
            self.trainer_cfg["ckpts_dir"] = os.path.join(dir_prefix,
                                                         str(n+1)+"/")

            trainer = Trainer(self.dataloader,
                              self.dataset_stats,
                              self.trainer_cfg,
                              self.optim_cfg,
                              self.model_cfg,
                              self.valid_dataloader,
                              self.lr_scheduler)
            trainer.train()

    def eval(self):
        pass
