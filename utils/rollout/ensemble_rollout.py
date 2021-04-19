import os
import torch
import numpy as np
from pathlib import Path
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from utils.visualization import EnsembleVisualize


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True


class EnsembleRollout(EnsembleVisualize):
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
        fname = Path(fname)
        dataset = self._read_single_demo(cfg["dataset"],
                                         [fname.name])

        loss, pred, gt_state, target = self._evaluate(dataset)
        recover_target = self._recover_data(gt_state, target)

        loss_stats = {"mean": loss, "std": np.zeros_like(loss)}
        pred_stats = {"mean": pred[0], "std": pred[1]}

        return loss_stats, pred_stats, recover_target, dataset.sample_time

    def _evaluate(self, dataset):
        cfg = deepcopy(self.cfg)
        dir_prefix = cfg["trainer"]["ckpts_dir"]

        rollout = self.horizon  # current rollout step
        ro_pred = None

        # get function handles of loss and metrics
        criterion = torch.nn.MSELoss(reduction='mean')
        losses = []
        pred_mean = []
        pred_std = []
        gt_states = []
        ro_states = []
        targets = []

        with torch.no_grad():
            for i in range(len(dataset)):
                state_action, target = dataset.__getitem__(i)
                gt_states.append(np.copy(state_action[:15]))
                targets.append(np.copy(target))

                if rollout < self.horizon:
                    state_action[:15] = np.copy(ro_pred)
                    rollout += 1
                else:
                    rollout = 0
                ro_states.append(np.copy(state_action[:15]))

                state_action = torch.tensor(state_action[None, ...]).to('cuda')
                target = torch.tensor(target[None, ...]).to('cuda')

                output_per_demo = []
                for n in range(cfg["trainer"]["num_ensemble"]):
                    cfg["eval"]["ckpt_dir"] = os.path.join(dir_prefix,
                                                           str(n+1)+"/")
                    model, _ = self._build_model(cfg)
                    output_per_model = model(state_action)
                    output_per_demo.append(output_per_model)
                output_per_demo = torch.cat(output_per_demo, dim=0)

                ro_pred = output_per_demo.cpu().numpy()
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
                losses.extend(loss)

        self.losses.extend(losses)
        return (np.array(losses),
                (np.array(pred_mean), np.array(pred_std)),
                np.array(gt_states), np.array(targets))
