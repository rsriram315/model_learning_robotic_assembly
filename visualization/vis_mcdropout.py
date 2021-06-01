import numpy as np
import torch
from copy import deepcopy
from pathlib import Path

from model import MCDropout
from dataloaders import Normalization
from visualization import EnsembleVisualize


class MCDropoutVisualize(EnsembleVisualize):
    def __init__(self, cfg, vis_dir="saved/visualizations"):
        self.cfg = cfg
        super().__init__(cfg, vis_dir)

    def visualize(self):
        # get the dataset stats
        cfg = deepcopy(self.cfg)

        for fname in self.demo_fnames:
            loss_stats, pred_stats, targets, time = \
                self.pred_stats(cfg, fname)

            if fname in self.train_demo_fnames:
                suffix_fname = Path('train') / Path(fname).stem
            elif fname in self.test_demo_fnames:
                suffix_fname = Path('test') / Path(fname).stem

            if self.vis_cfg["loss"]:
                loss_fname = self.vis_dir / "loss" / suffix_fname
                self._vis_loss(loss_stats["mean"],
                               loss_stats["std"],
                               time, loss_fname)

            if self.vis_cfg["axis"]:
                axis_fname = self.vis_dir / "axis" / suffix_fname
                self._vis_axis(pred_stats["mean"][:-1, :],
                               pred_stats["std"][:-1, :],
                               targets[1:, :],
                               time[1:],
                               axis_fname)

            if self.vis_cfg["trajectory"]:
                traj_fname = self.vis_dir / "trajectory" / suffix_fname
                self._vis_trajectory(pred_stats["mean"][:-1, :3],
                                     targets[1:, :3],
                                     traj_fname)

            print(f"... Generated visualization for {suffix_fname.name}\n")

    def pred_stats(self, cfg, fname):
        losses_per_demo = []
        preds_per_demo = []
        model, cfg = self._build_model(cfg)
        self.norm = Normalization(cfg["dataset"]["stats"])

        fname = Path(fname)
        dataset = self._read_single_demo(cfg["dataset"],
                                         [fname.name])

        for _ in range(cfg["eval"]["num_mc"]):
            loss, pred, state = self._evaluate(model, dataset)

            recover_pred, recover_target = self._recover_data(pred, state)

            losses_per_demo.append(loss)
            preds_per_demo.append(recover_pred)

        loss_stats = {"mean": np.mean(losses_per_demo, axis=0),
                      "std": np.std(losses_per_demo, axis=0)}
        pred_stats = {"mean": np.mean(preds_per_demo, axis=0),
                      "std": np.std(preds_per_demo, axis=0)}

        return loss_stats, pred_stats, recover_target, dataset.sample_time

    def enable_dropout(self, m):
        for each_module in m.modules():
            if each_module.__class__.__name__.startswith('Dropout'):
                each_module.train()

    def _build_model(self, cfg):
        if cfg["eval"]["ckpt_pth"] is None:
            ckpt_pth = self._find_ckpt(cfg["eval"]["ckpt_dir"])
        else:
            ckpt_pth = cfg["eval"]["ckpt_pth"]

        ckpt = torch.load(ckpt_pth)
        model_cfg = cfg["model"]
        cfg["dataset"]["stats"] = ckpt["dataset_stats"]

        # build model architecture, then print to console
        model = MCDropout(model_cfg["input_dims"], model_cfg["output_dims"])
        model.load_state_dict(ckpt["state_dict"])

        model = model.to(self.device)
        if len(self.device_ids) > 1:
            model = torch.nn.DataParallel(model, self.device_ids)
        model.eval()
        self.enable_dropout(model)

        return model, cfg
