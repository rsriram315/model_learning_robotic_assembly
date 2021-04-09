import torch
import numpy as np
from copy import deepcopy
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from dataloaders import Normalization
from visualize import MCDropoutVisualize


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True


class MCRollout(MCDropoutVisualize):
    def __init__(self, cfg, horizon):
        super().__init__(cfg, vis_dir=f"saved/visualizations_{horizon}")
        self.horizon = horizon
        self.losses = 0

    def pred_stats(self):
        cfg = deepcopy(self.cfg)
        model, ds_stats = self._build_model(cfg)
        self.norm = Normalization(ds_stats)

        loss_mean = []
        loss_std = []
        pred_mean = []
        pred_std = []
        targets = []
        time = []

        for fname in self.demo_fnames:
            fname = Path(fname)
            dataset = self._read_single_demo(cfg["dataset"],
                                             [fname.name], ds_stats)
            time.append(dataset.sample_time)

            losses_per_demo, preds_mean, preds_std, target_per_demo = \
                self._evaluate(model, dataset)

            self.losses += np.sum(losses_per_demo)

            loss_mean.append(losses_per_demo)
            loss_std.append(np.zeros_like(losses_per_demo))
            pred_mean.append(preds_mean)
            pred_std.append(preds_std)
            targets.append(target_per_demo)

        loss_stats = {"mean": loss_mean, "std": loss_std}
        pred_stats = {"mean": pred_mean, "std": pred_std}

        return loss_stats, pred_stats, targets, time

    def _evaluate(self, model, dataset):
        rollout = self.horizon  # current rollout step
        rollout_pred = None

        # get function handles of loss and metrics
        criterion = torch.nn.MSELoss(reduction='sum')
        losses = []
        preds_mean = []
        preds_std = []
        targets = []
        num_mc = self.cfg["eval"]["num_mc"]

        with torch.no_grad():
            for i in range(len(dataset)):
                s_a, target = dataset.__getitem__(i)
                state_orig = np.copy(s_a[:15])

                if rollout < self.horizon:
                    s_a[:15] = np.copy(rollout_pred)
                    rollout += 1
                else:
                    rollout = 0

                # expand to num_mc x feature_dim
                state_action = np.vstack([s_a for _ in range(num_mc)])
                state_action = torch.tensor(state_action).to('cuda')
                target = torch.tensor(target[None, ...]).to('cuda')

                output = model(state_action)

                recover_res = \
                    self.norm.res_inv_normalize(output.cpu().numpy())
                target_res = \
                    self.norm.res_inv_normalize(target.cpu().numpy())
                recover_state = \
                    self.norm.inv_normalize(state_orig[None, None, :])
                recover_output = recover_res + recover_state
                recover_target = target_res + recover_state

                recover_mean_output = np.mean(recover_output, axis=0,
                                              keepdims=True)
                rollout_pred = self.norm.normalize(
                                recover_mean_output[:, None, :])

                loss = criterion(torch.Tensor(recover_mean_output),
                                 torch.Tensor(recover_target))
                # output euler angles
                pred_angle = \
                    (R.from_matrix(recover_output[:, 6:].reshape(-1, 3, 3))
                      .as_euler('xyz', degrees=True))
                target_angle = \
                    (R.from_matrix(recover_target[:, 6:].reshape(-1, 3, 3))
                      .as_euler('xyz', degrees=True))

                preds = np.hstack((recover_output, pred_angle))
                preds_mean.append(np.mean(preds, axis=0))
                preds_std.append(np.std(preds, axis=0))
                targets.extend(np.hstack((recover_target, target_angle)))
                losses.append(loss.item())

        return (np.array(losses), np.array(preds_mean), np.array(preds_std),
                np.array(targets))
