import torch
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from visualize import MCDropoutVisualize
from dataloaders import Normalization


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True


class MCRollout(MCDropoutVisualize):
    def __init__(self, cfg, horizon):
        super().__init__(cfg, vis_dir=f"saved/visualizations_{horizon}")
        self.horizon = horizon
        self.losses = []

    def _recover_data(self, state, target):
        curr_state = self.norm.inv_normalize(state[:, None, :])
        recover_target = self.norm.inv_normalize(target[:, None, :],
                                                 is_res=True)
        recover_target += curr_state

        # output euler angles
        target_angle = (R.from_matrix(recover_target[:, 6:].reshape(-1, 3, 3))
                         .as_euler('xyz', degrees=True))

        recover_target = np.hstack((recover_target, target_angle))
        return recover_target

    def pred_stats(self, cfg, fname):
        model, cfg = self._build_model(cfg)
        self.norm = Normalization(cfg["dataset"]["stats"])

        fname = Path(fname)
        dataset = self._read_single_demo(cfg["dataset"],
                                         [fname.name])

        loss, pred, gt_state, target = \
            self._evaluate(model, dataset)
        recover_targets = self._recover_data(gt_state, target)

        loss_stats = {"mean": loss, "std": np.zeros_like(loss)}
        pred_stats = {"mean": pred[0], "std": pred[1]}

        return loss_stats, pred_stats, recover_targets, dataset.sample_time

    def _evaluate(self, model, dataset):
        rollout = self.horizon  # current rollout step
        ro_pred = None

        # get function handles of loss and metrics
        criterion = torch.nn.MSELoss(reduction='none')
        num_mc = self.cfg["eval"]["num_mc"]
        losses = []
        pred_mean = []
        pred_std = []
        gt_states = []
        ro_states = []
        targets = []

        with torch.no_grad():
            for i in range(len(dataset)):
                s_a, target = dataset.__getitem__(i)
                gt_states.append(np.copy(s_a[:15]))
                targets.append(np.copy(target))

                if rollout < self.horizon:
                    s_a[:15] = np.copy(ro_pred)
                    rollout += 1
                else:
                    rollout = 0
                ro_states.append(np.copy(s_a[:15]))

                # expand to num_mc x feature_dim
                state_action = np.vstack([s_a for _ in range(num_mc)])
                state_action = torch.tensor(state_action).to('cuda')
                target = torch.tensor(target[None, ...]).to('cuda')

                output = model(state_action)

                ro_pred = output.cpu().numpy()
                ro_pred = self.norm.inv_normalize(ro_pred[:, None, :],
                                                  is_res=True)
                ro_pred += self.norm.inv_normalize(ro_states[-1]
                                                   [None, None, :])

                # record mean and standard deviation
                # output euler angles
                pred_angle = (R.from_matrix(ro_pred[:, 6:].reshape(-1, 3, 3))
                               .as_euler('xyz', degrees=True))
                prediction = np.hstack((ro_pred, pred_angle))
                pred_mean.append(np.mean(prediction, axis=0))
                pred_std.append(np.std(prediction, axis=0))

                ro_pred = self.norm.normalize(
                    np.mean(ro_pred, axis=0, keepdims=True)[:, None, :])

                loss = criterion(torch.tensor(ro_pred),
                                 torch.tensor(gt_states[-1][None, ...]))
                losses.extend(torch.sum(loss, dim=1))

        self.losses.extend(losses)
        return (np.array(losses), (np.array(pred_mean), np.array(pred_std)),
                np.array(gt_states), np.array(targets))
