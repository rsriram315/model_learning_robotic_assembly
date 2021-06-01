import torch
import numpy as np
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
from functools import partial

from model import MLP, MCDropout
from dataloaders import DemoDataset, DemoDataLoader
from utils.logger import write_log
from utils import prepare_device

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True


class Evaluate:
    def __init__(self, cfg):
        self.cfg = deepcopy(cfg)
        self.eval_log = partial(write_log, self.cfg["eval"]["log_file"])

        self.device, self.device_ids = prepare_device(
                                        self.cfg["model"]["n_gpu"])

        if self.cfg["eval"]["ckpt_pth"] is None:
            ckpt_pth = self._find_ckpt(self.cfg["eval"]["ckpt_dir"])
        else:
            ckpt_pth = self.cfg["eval"]["ckpt_pth"]

        self.model, self.cfg = self._build_model(self.cfg,
                                                 ckpt_pth)

        self.dataloader, self.demo_fnames =\
            self._load_demos(self.cfg["dataset"], self.cfg["dataloader"])

    def evaluate(self):
        criterion = torch.nn.MSELoss(reduction='mean')

        losses = []
        with torch.no_grad():
            for state_action, target in tqdm(self.dataloader):
                state_action, target = (state_action.to(self.device),
                                        target.to(self.device))
                output = self.model(state_action)

                loss = criterion(output, target)
                losses.append(loss.item())

        self.eval_log(f"... Total test samples is "
                      f"{len(self.dataloader.sampler)}")
        self.eval_log(f'... Test loss: {np.mean(losses)}')

    def _load_demos(self, ds_cfg, dl_cfg):
        # for demos for test set
        if len(ds_cfg["fnames"]) == 0:
            ds_root = Path(ds_cfg["root"])
            demos = [pth.name for pth in list(ds_root.glob("*.h5"))]
        else:
            demos = ds_cfg["fnames"]

        num_train_demo = int(len(demos) * 0.8)
        ds_cfg["fnames"] = (np.random.RandomState(ds_cfg["seed"])
                              .permutation(demos)[num_train_demo:])

        dataset = DemoDataset(ds_cfg)
        dataloader = DemoDataLoader(dataset, dl_cfg)

        self.eval_log("... Holdout demos are:\n")
        for demo in dataloader.get_fnames():
            self.eval_log(f"   {demo}\n")
        return dataloader, ds_cfg["fnames"]

    def _find_ckpt(self, ckpt_dir):
        ckpt_dir = Path(ckpt_dir)
        ckpt_pths = [pth for pth in list(ckpt_dir.glob("*.pth"))]
        return ckpt_pths[0]

    def _build_model(self, cfg, ckpt_pth):
        ckpt = torch.load(ckpt_pth)

        cfg["dataset"]["stats"] = ckpt["dataset_stats"]
        model_cfg = cfg["model"]

        # build model architecture, then print to console
        if model_cfg["name"] == "MLP":
            model = MLP(model_cfg["input_dims"],
                        model_cfg["output_dims"])
        elif model_cfg["name"] == "MCDropout":
            model = MCDropout(model_cfg["input_dims"],
                              model_cfg["output_dims"])
        print(model)

        # load model checkpoint
        model.load_state_dict(ckpt["state_dict"])
        self.eval_log(f'... Load checkpoint: {ckpt_pth}')

        model = model.to(self.device)
        if len(self.device_ids) > 1:
            model = torch.nn.DataParallel(model, self.device_ids)
        model.eval()

        return model, cfg
