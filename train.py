import argparse
import torch
import numpy as np
from model import MLP
from trainer import Trainer
from dataloaders import DemoDataset, DemoDataLoader
from utils import prepare_device, read_json


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
np.random.seed(SEED)


def train(cfg_path):
    # setup dataloader instances
    cfg = read_json(cfg_path)
    dataset_cfg = cfg["dataset"]
    dataloader_cfg = cfg["dataloader"]
    dataset = DemoDataset(**dataset_cfg["params"])
    train_dataset = dataset.split_train_test(train=True,
                                             seed=dataset_cfg["seed"])
    dataloader = DemoDataLoader(train_dataset, dataloader_cfg)

    valid_dataloader = dataloader.split_validation()

    print(f"... {dataloader.n_samples} training samples")

    # build model architecture, then print to console
    model = MLP(input_dims=12, output_dims=6)
    print(model)
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(cfg["n_gpu"])
    model = model.to(device)

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = torch.nn.MSELoss()
    # TODO: metrics
    metrics = []

    # build optimizer, learning rate scheduler. delete every lines containing
    # lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=0.001)

    trainer = Trainer(model, criterion, metrics, optimizer, **cfg["trainer"],
                      device=device, dataloader=dataloader,
                      valid_dataloader=valid_dataloader)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    args = parser.parse_args()
    train(args.config)
