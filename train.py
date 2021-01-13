import argparse
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)

def main(config):
    # TODO: setup logger

    # TODO: setup data_loader instances

    # TODO: build model architecture

    trainer = Trainer(model, criterion, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader)
    trainer.train()