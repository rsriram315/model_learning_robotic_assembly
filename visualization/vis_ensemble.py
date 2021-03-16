import setup  # noqa
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .vis_mlp import Visualize
from dataloaders import Normalization, Standardization


class EnsembleVisualize(Visualize):
    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__(cfg)

    def visualize(self):
        loss_stats, pred_stats, ds_stats = self.pred_stats()

        for i in range(len(self.demo_fnames)):
            if self.demo_fnames[i] in self.train_demo_fnames:
                suffix_fname = Path('train') / Path(self.demo_fnames[i]).stem
            elif self.demo_fnames[i] in self.test_demo_fnames:
                suffix_fname = Path('test') / Path(self.demo_fnames[i]).stem

            dataset = self._read_single_demo([self.demo_fnames[i]], ds_stats)

            time = dataset.sample_time
            state = dataset.states_actions[:, 0]
            state = self.norm.inverse_normalize(state, is_target=True)

            if self.vis_cfg["loss"]:
                loss_fname = self.vis_dir / "loss" / suffix_fname
                self._vis_loss(loss_stats["mean"][i],
                               loss_stats["std"][i],
                               time, loss_fname)

            if self.vis_cfg["axis"]:
                axis_fname = self.vis_dir / "axis" / suffix_fname
                self._vis_axis(pred_stats["mean"][i],
                               pred_stats["std"][i],
                               state, time, axis_fname)

            if self.vis_cfg["trajectory"]:
                traj_fname = self.vis_dir / "trajectory" / suffix_fname
                self._vis_trajectory(pred_stats["mean"][i][:, :3],
                                     state[:, :3], traj_fname)

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
        plt.savefig(fname)
        plt.close(fig)

    def _vis_axis(self, pred_mean, pred_std, state, time, fname):
        fig, axs = plt.subplots(3, 3, figsize=(30, 15), sharex='all')

        # the predicted states should start with the prediction for t=1 not t=0
        size = 2
        features = ['pos', 'rot', 'force']
        axis = ['x', 'y', 'z']

        for r, feature in enumerate(features):
            for c, ax in enumerate(axis):
                idx = c + 3 * r

                axs[r, c].plot(time[1:], pred_mean[:-1, idx],
                               c='tab:orange', label='mean predictions')
                axs[r, c].fill_between(time[1:],
                                       (pred_mean[:-1, idx] +
                                        2 * pred_std[:-1, idx]),
                                       (pred_mean[:-1, idx] -
                                        2 * pred_std[:-1, idx]),
                                       alpha=0.5, color='tab:orange',
                                       label='confidence interval')

                axs[r, c].scatter(time, state[:, idx], s=size,
                                  c='tab:blue', label='ground truth')
                axs[r, c].set_title(f'{feature} {ax} axis')
                axs[r, c].set_ylabel('coordinate')
                if r == 2:
                    axs[r, c].set_xlabel('time')

        plt.savefig(fname)
        plt.close(fig)

    def pred_stats(self):
        num_ensemble = self.cfg["trainer"]["num_ensemble"]
        dir_prefix = self.cfg["trainer"]["ckpts_dir"]

        losses = []
        preds = []

        for n in range(num_ensemble):
            self.cfg["eval"]["ckpt_dir"] = os.path.join(dir_prefix,
                                                        str(n+1)+"/")
            model, ds_stats = self._build_model(self.cfg)

            if self.cfg["dataset"]["preprocess"]["normalize"]:
                self.norm = Normalization(ds_stats)
            elif self.cfg["dataset"]["preprocess"]["standardize"]:
                self.norm = Standardization(ds_stats)

            losses_per_model = []
            preds_per_model = []
            demo_lens = []

            for fname in self.demo_fnames:
                fname = Path(fname)
                dataset = self._read_single_demo([fname.name], ds_stats)
                losses_per_demo, preds_per_demo = self._evaluate(model,
                                                                 dataset)
                demo_lens.append(len(dataset))

                losses_per_model.extend(losses_per_demo)
                preds_per_model.extend(preds_per_demo)

            losses.append(losses_per_model)
            preds.append(preds_per_model)

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

    def slice_arr(self, demo_lens, arr):
        res = []
        start = 0

        for end in demo_lens:
            res.append(arr[start:start + end])
            start += end
        return np.array(res, dtype=object)
