import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path

from visualization.base_vis import BaseVisualize
from dataloaders import Normalization


class EnsembleVisualize(BaseVisualize):
    def __init__(self, cfg, vis_dir="saved/visualizations"):
        super().__init__(cfg, vis_dir)

    def visualize(self):
        # get the dataset stats
        cfg = deepcopy(self.cfg)
        cfg["eval"]["ckpt_dir"] = \
            os.path.join(cfg["trainer"]["ckpts_dir"], str(1)+"/")

        _, cfg = self._build_model(cfg)
        self.norm = Normalization(cfg["dataset"]["stats"])

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

            print(f"... Generated visualization for {suffix_fname.name}")

    def _vis_loss(self, loss_mean, loss_std, time, fname):
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.set_title(f"mean of the loss: {np.mean(loss_mean)}")

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

    def _vis_axis(self, pred_mean, pred_std, target,
                  time, fname, plot_mat=False):
        size = 1
        if plot_mat:
            features = ['pos', 'force', 'matrix R row 1', 'matrix R row 2',
                        'matrix R row 3', 'euler angles']
            figsize = (25, 20)
        else:
            features = ['pos', 'force', 'euler angles']
            figsize = (20, 10)
        axis = ['x', 'y', 'z']

        rows = len(features)
        cols = len(axis)
        fig, axs = plt.subplots(rows, cols, figsize=figsize, sharex='all')

        for r, feature in enumerate(features):
            # not plotting matrix, skip the matrix
            if not plot_mat:
                r = 5 if r > 2 else r
                feature = features[r]
            for c, ax in enumerate(axis):
                idx = c + 3 * r

                axs[r, c].plot(time, pred_mean[:, idx],
                               c='tab:orange', label='mean predictions')

                upper_ci = pred_mean[:, idx] + 2 * pred_std[:, idx]
                lower_ci = pred_mean[:, idx] - 2 * pred_std[:, idx]
                # fixed the angle confidence interval exceeds [-180, 180]
                if idx >= 11:
                    upper_ci = np.clip(upper_ci, -185, 185)
                    lower_ci = np.clip(lower_ci, -185, 185)
                axs[r, c].fill_between(time,
                                       upper_ci,
                                       lower_ci,
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

    def pred_stats(self, cfg, fname):
        dir_prefix = cfg["trainer"]["ckpts_dir"]
        losses_per_demo = []
        preds_per_demo = []

        fname = Path(fname)
        dataset = self._read_single_demo(cfg["dataset"],
                                         [fname.name])

        for n in range(cfg["trainer"]["num_ensemble"]):
            cfg["eval"]["ckpt_dir"] = os.path.join(dir_prefix,
                                                   str(n+1)+"/")
            model, _ = self._build_model(cfg)

            loss, pred, state = self._evaluate(model, dataset)

            recover_pred, recover_target = self._recover_data(pred, state)

            losses_per_demo.append(loss)
            preds_per_demo.append(recover_pred)

        loss_stat = {"mean": np.mean(losses_per_demo, axis=0),
                     "std": np.std(losses_per_demo, axis=0)}
        pred_stat = {"mean": np.mean(preds_per_demo, axis=0),
                     "std": np.std(preds_per_demo, axis=0)}

        return loss_stat, pred_stat, recover_target, dataset.sample_time
