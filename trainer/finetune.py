import torch
import numpy as np
from copy import deepcopy
from pathlib import Path

from trainer.finetune_trainer import Trainer
from trainer.ensemble_trainer import EnsembleTrainer
from dataloaders import DemoDataLoader
from dataloaders.dataset_panda import DemoDataset


# # fix random seeds for reproducibility
# SEED = 123
# torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# # torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
# np.random.seed(SEED)


def finetune(cfg):
    cfg = deepcopy(cfg)
    # for demos for training set
    if len(cfg["finetune_dataset"]["fnames"]) == 0:
        ds_root = Path(cfg["finetune_dataset"]["root"])
        demos = [pth.name for pth in list(ds_root.glob("*.h5"))]
    else:
        demos = cfg["finetune_dataset"]["fnames"]

    num_train_demo = int(len(demos) * 0.8)
    cfg["finetune_dataset"]["fnames"] = \
        (np.random.RandomState(cfg["finetune_dataset"]["seed"])
           .permutation(demos)[:num_train_demo])

    dataset = DemoDataset(cfg["finetune_dataset"], is_train=True)
    dataloader = DemoDataLoader(dataset, cfg["dataloader"])
    valid_dataloader = dataloader.split_validation()
    # print(f"... {dataloader.n_samples} training samples")

    trainer_name = cfg["finetune_trainer"]["name"]
    if trainer_name == "finetune_mlp":
        trainer = Trainer(dataloader,
                          dataset.stats,
                          cfg["finetune_dataset"],
                          cfg["finetune_trainer"],
                          cfg["finetune_optimizer"],
                          cfg["model"],
                          resume_path=cfg["trainer"]["ckpts_dir"],
                          valid_dataloader=valid_dataloader)
    elif trainer_name == "ensemble":
        trainer = EnsembleTrainer(dataloader,
                                  dataset.stats,
                                  cfg["finetune_dataset"],
                                  cfg["trainer"],
                                  cfg["optimizer"],
                                  cfg["model"],
                                  valid_dataloader=valid_dataloader)

    trainer.train()
