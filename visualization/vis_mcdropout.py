import setup  # noqa
import numpy as np
import torch
from pathlib import Path
from .vis_ensemble import EnsembleVisualize
from dataloaders import Normalization, Standardization
from model import MCDropout


class MCDropoutVisualize(EnsembleVisualize):
    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__(cfg)

    def pred_stats(self):
        num_mc = self.cfg["trainer"]["num_mc"]
        model, ds_stats = self._build_model(self.cfg)
        self.enable_dropout(model)

        if self.cfg["dataset"]["preprocess"]["normalize"]:
            self.norm = Normalization(ds_stats)
        elif self.cfg["dataset"]["preprocess"]["standardize"]:
            self.norm = Standardization(ds_stats)

        losses = []
        preds = []

        for _ in range(num_mc):
            losses_per_sample = []
            preds_per_sample = []
            demo_lens = []

            for fname in self.demo_fnames:
                fname = Path(fname)
                dataset = self._read_single_demo([fname.name], ds_stats)
                losses_per_demo, preds_per_demo = self._evaluate(model,
                                                                 dataset)
                demo_lens.append(len(dataset))

                losses_per_sample.extend(losses_per_demo)
                preds_per_sample.extend(preds_per_demo)

            losses.append(losses_per_sample)
            preds.append(preds_per_sample)

        demo_lens = demo_lens[:len(self.demo_fnames)]
        loss_mean = self.slice_arr(demo_lens, np.mean(losses, axis=0))
        loss_std = self.slice_arr(demo_lens, np.std(losses, axis=0))
        loss_stats = {"mean": loss_mean,
                      "std": loss_std}

        pred_mean = self.slice_arr(demo_lens, np.mean(preds, axis=0))
        pred_std = self.slice_arr(demo_lens, np.std(preds, axis=0))
        pred_stats = {"mean": pred_mean,
                      "std": pred_std}

        return loss_stats, pred_stats, ds_stats

    def enable_dropout(self, m):
        for each_module in m.modules():
            if each_module.__class__.__name__.startswith('Dropout'):
                each_module.train()

    def _build_model(self, cfg):
        if cfg["eval"]["ckpt_pth"] is None:
            ckpt_pth = self._find_ckpt(cfg["eval"]["ckpt_dir"])
        else:
            ckpt_pth = cfg["eval"]["ckpt_pth"]

        model_cfg = cfg["model"]

        # build model architecture, then print to console
        model = MCDropout(model_cfg["input_dims"], model_cfg["output_dims"])

        ckpt = torch.load(ckpt_pth)
        model.load_state_dict(ckpt["state_dict"])

        model = model.to(self.device)
        if len(self.device_ids) > 1:
            model = torch.nn.DataParallel(model, self.device_ids)
        model.eval()

        dataset_stats = ckpt["dataset_stats"]
        return model, dataset_stats
