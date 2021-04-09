import torch
import numpy as np
from copy import deepcopy
from pathlib import Path
from trainer import Trainer, EnsembleTrainer
from dataloaders import DemoDataset, DemoDataLoader


# # fix random seeds for reproducibility
# SEED = 123
# torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# # torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
# np.random.seed(SEED)


def train(cfg):
    cfg = deepcopy(cfg)
    # for demos for training set
    if len(cfg["dataset"]["fnames"]) == 0:
        ds_root = Path(cfg["dataset"]["root"])
        demos = [pth.name for pth in list(ds_root.glob("*.h5"))]
    else:
        demos = cfg["dataset"]["fnames"]

    num_train_demo = int(len(demos) * 0.8)
    cfg["dataset"]["fnames"] = \
        (np.random.RandomState(cfg["dataset"]["seed"])
           .permutation(demos)[:num_train_demo])

    dataset = DemoDataset(cfg["dataset"])
    dataloader = DemoDataLoader(dataset, cfg["dataloader"])
    valid_dataloader = dataloader.split_validation()
    print(f"... {dataloader.n_samples} training samples")

    trainer_name = cfg["trainer"]["name"]
    if trainer_name == "mlp":
        trainer = Trainer(dataloader,
                          dataset.stats,
                          cfg["trainer"],
                          cfg["optimizer"],
                          cfg["model"],
                          valid_dataloader=valid_dataloader)
    elif trainer_name == "ensemble":
        trainer = EnsembleTrainer(dataloader,
                                  dataset.stats,
                                  cfg["trainer"],
                                  cfg["optimizer"],
                                  cfg["model"],
                                  valid_dataloader=valid_dataloader)

    trainer.train()
