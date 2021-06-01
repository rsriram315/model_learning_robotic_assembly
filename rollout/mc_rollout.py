import torch
import numpy as np
from pathlib import Path
from copy import deepcopy

from visualization import MCDropoutVisualize
from dataloaders import Normalization, recover_rotation, add_euler_angle


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True


class MCRollout(MCDropoutVisualize):
    def __init__(self, cfg, horizon):
        super().__init__(cfg, vis_dir=f"saved/visualizations_{horizon}")
        self.horizon = horizon
        self.losses = []

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
                               time[1:], loss_fname)

            if self.vis_cfg["axis"]:
                axis_fname = self.vis_dir / "axis" / suffix_fname
                self._vis_axis(pred_stats["mean"][:-1, :],
                               pred_stats["std"][:-1, :],
                               targets[1:, :],
                               time[1:-1],
                               axis_fname)

            if self.vis_cfg["trajectory"]:
                traj_fname = self.vis_dir / "trajectory" / suffix_fname
                self._vis_trajectory(pred_stats["mean"][:-1, :3],
                                     targets[1:, :3],
                                     traj_fname)

            print(f"... Generated visualization for {suffix_fname.name}\n")

    def pred_stats(self, cfg, fname):
        model, cfg = self._build_model(cfg)
        self.norm = Normalization(cfg["dataset"]["stats"])

        fname = Path(fname)
        dataset = self._read_single_demo(cfg["dataset"],
                                         [fname.name])

        loss, pred, gt_states = \
            self._evaluate(model, dataset)

        loss_stats = {"mean": loss, "std": np.zeros_like(loss)}
        pred_stats = {"mean": pred[0], "std": pred[1]}

        return loss_stats, pred_stats, gt_states, dataset.sample_time

    def _evaluate(self, model, dataset):
        rollout = self.horizon  # current rollout step
        ro_pred = None

        # get function handles of loss and metrics
        criterion = torch.nn.MSELoss(reduction='mean')
        num_mc = self.cfg["eval"]["num_mc"]
        losses = []
        pred_mean = []
        pred_std = []
        gt_states = []
        ro_states = []

        with torch.no_grad():
            for i in range(len(dataset) - 1):
                s_a, target = dataset.__getitem__(i)
                gt_state = np.copy(s_a[:15])  # ground truth states

                next_state_action, _ = dataset.__getitem__(i+1)
                next_gt_state = np.copy(next_state_action[None, :15])

                if rollout < self.horizon:
                    s_a[:15] = np.copy(ro_pred)
                    rollout += 1
                else:
                    rollout = 0
                ro_state = np.copy(s_a[:15])
                ro_states.append(ro_state)

                # expand to num_mc x feature_dim
                state_action = np.vstack([s_a for _ in range(num_mc)])
                state_action = torch.tensor(state_action).to('cuda')
                target = torch.tensor(target[None, ...]).to('cuda')

                output = model(state_action)

                # num_mc predictions
                ro_pred = output.cpu().numpy()
                ro_state_stacked = np.vstack([ro_state for _ in range(num_mc)])
                gt_state_stacked = np.vstack([gt_state for _ in range(num_mc)])

                recover_ro_pred, recover_gt_state = \
                    self._recover_data(ro_pred,
                                       ro_state_stacked,
                                       gt_state_stacked)

                gt_states.append(recover_gt_state[0, :])
                # record mean and standard deviation
                pred_mean.append(np.mean(recover_ro_pred, axis=0))
                pred_std.append(np.std(recover_ro_pred, axis=0))

                ro_pred = self.norm.normalize(pred_mean[-1][None, None, :15])

                loss = criterion(torch.tensor(ro_pred),
                                 torch.tensor(next_gt_state))
                losses.append(loss.item())

        self.losses.extend(losses)
        return (np.array(losses), (np.array(pred_mean), np.array(pred_std)),
                np.array(gt_states))

    def _recover_data(self, ro_pred, ro_state, gt_state):
        curr_ro_state = self.norm.inv_normalize(ro_state[:, None, :])
        recover_ro_output = self.norm.inv_normalize(ro_pred[:, None, :],
                                                    is_res=True)
        recover_ro_output[:, :6] += curr_ro_state[:, :6]
        recover_ro_output = recover_rotation(recover_ro_output, curr_ro_state)
        recover_ro_output = add_euler_angle(recover_ro_output)

        recover_gt_state = self.norm.inv_normalize(gt_state[:, None, :])
        recover_gt_state = add_euler_angle(recover_gt_state)
        return recover_ro_output, recover_gt_state
