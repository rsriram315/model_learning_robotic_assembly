import setup  # noqa
import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path
from .vis_mlp import Visualize
from dataloaders import Normalization, Standardization


class EnsembleVisualize(Visualize):
    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__(cfg)

    def visualize(self):
        loss_stats, pred_stats, targets, time = self.pred_stats()

        for i in range(len(self.demo_fnames)):
            if self.demo_fnames[i] in self.train_demo_fnames:
                suffix_fname = Path('train') / Path(self.demo_fnames[i]).stem
            elif self.demo_fnames[i] in self.test_demo_fnames:
                suffix_fname = Path('test') / Path(self.demo_fnames[i]).stem

            if self.vis_cfg["loss"]:
                loss_fname = self.vis_dir / "loss" / suffix_fname
                self._vis_loss(loss_stats["mean"][i],
                               loss_stats["std"][i],
                               time[i], loss_fname)

            if self.vis_cfg["axis"]:
                axis_fname = self.vis_dir / "axis" / suffix_fname
                self._vis_axis(pred_stats["mean"][i],
                               pred_stats["std"][i],
                               targets[i], time[i], axis_fname)

            if self.vis_cfg["trajectory"]:
                traj_fname = self.vis_dir / "trajectory" / suffix_fname
                self._vis_trajectory(pred_stats["mean"][i][:, :3],
                                     targets[i][:, :3], traj_fname)

            print(f"... Generated visualization for {suffix_fname.name}")

    def _vis_loss(self, loss_mean, loss_std, time, fname):
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.set_title(f"sum of the loss: {sum(loss_mean)}")

        ax.plot(time, loss_mean)
        ax.fill_between(time,
                        loss_mean + 2 * loss_std,
                        loss_mean - 2 * loss_std,
                        alpha=0.5, label='confidence interval')

        ax.set_xlabel("time")
        ax.set_ylabel("loss")
        ax.legend()

        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close(fig)

    def _vis_axis(self, pred_mean, pred_std, target, time, fname):
        size = 1
        # features = ['pos', 'force', 'rot_cosine', 'rot_sine']
        features = ['pos', 'force', 'matrix R row 1', 'matrix R row 2',
                    'matrix R row 3', 'euler angles']
        axis = ['x', 'y', 'z']

        rows = len(features)
        cols = len(axis)
        fig, axs = plt.subplots(rows, cols, figsize=(30, 30), sharex='all')

        for r, feature in enumerate(features):
            for c, ax in enumerate(axis):
                idx = c + 3 * r

                axs[r, c].plot(time, pred_mean[:, idx],
                               c='tab:orange', label='mean predictions')
                axs[r, c].fill_between(time,
                                       (pred_mean[:, idx] +
                                        2 * pred_std[:, idx]),
                                       (pred_mean[:, idx] -
                                        2 * pred_std[:, idx]),
                                       alpha=0.5, color='tab:orange',
                                       label='confidence interval')

                axs[r, c].scatter(time, target[:, idx], s=size,
                                  c='tab:blue', label='ground truth')
                axs[r, c].set_title(f'{feature} {ax} axis')
                axs[r, c].set_ylabel('coordinate')
                if r == rows - 1:
                    axs[r, c].set_xlabel('time')

        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close(fig)

    def pred_stats(self):
        num_ensemble = self.cfg["trainer"]["num_ensemble"]
        dir_prefix = self.cfg["trainer"]["ckpts_dir"]

        loss_mean = []
        loss_std = []
        pred_mean = []
        pred_std = []
        targets = []
        time = []

        # get the dataset stats
        cfg = deepcopy(self.cfg)
        cfg["eval"]["ckpt_dir"] = os.path.join(dir_prefix,
                                               str(1)+"/")
        _, ds_stats = self._build_model(cfg)

        if self.cfg["dataset"]["preprocess"]["normalize"]:
            self.norm = Normalization(ds_stats)
        elif self.cfg["dataset"]["preprocess"]["standardize"]:
            self.norm = Standardization(ds_stats)

        for fname in self.demo_fnames:
            losses_per_demo = []
            preds_per_demo = []

            fname = Path(fname)
            dataset = self._read_single_demo(deepcopy(self.cfg["dataset"]),
                                             [fname.name], ds_stats)
            time.append(dataset.sample_time)

            for n in range(num_ensemble):
                cfg["eval"]["ckpt_dir"] = os.path.join(dir_prefix,
                                                       str(n+1)+"/")
                model, ds_stats = self._build_model(cfg)

                losses_per_model, preds_per_model, target_per_model = \
                    self._evaluate(model, dataset)

                losses_per_demo.append(losses_per_model)
                preds_per_demo.append(preds_per_model)
                if n == 0:
                    targets.append(target_per_model)

            loss_mean.append(np.mean(losses_per_demo, axis=0))
            loss_std.append(np.std(losses_per_demo, axis=0))
            pred_mean.append(np.mean(preds_per_demo, axis=0))
            pred_std.append(np.std(preds_per_demo, axis=0))

        loss_stats = {"mean": loss_mean, "std": loss_std}
        pred_stats = {"mean": pred_mean, "std": pred_std}

        return loss_stats, pred_stats, targets, time
