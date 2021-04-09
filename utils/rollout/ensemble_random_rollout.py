import os
import torch
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from .mlp_rollout import Rollout


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True


class EnsembleRandomRollout(Rollout):
    def __init__(self, cfg, horizon):
        # a initial ckpt path is needed
        init_cfg = deepcopy(cfg)
        init_cfg["eval"]["ckpt_dir"] = "saved/ckpts/1/"
        super().__init__(init_cfg, horizon)

    def _evaluate(self, model, dataset):
        rollout = self.horizon  # current rollout step
        rollout_pred = None

        cfg = deepcopy(self.cfg)
        num_ensemble = cfg["trainer"]["num_ensemble"]
        dir_prefix = cfg["trainer"]["ckpts_dir"]

        # get function handles of loss and metrics
        criterion = torch.nn.MSELoss(reduction='sum')
        losses = []
        preds = []
        targets = []

        with torch.no_grad():
            for i in range(len(dataset)):
                state_action, target = dataset.__getitem__(i)
                state_orig = np.copy(state_action[:15])

                if rollout < self.horizon:
                    state_action[:15] = rollout_pred
                    rollout += 1
                else:
                    rollout = 0

                state_action = torch.tensor(state_action[None, ...]).to('cuda')
                target = torch.tensor(target[None, ...]).to('cuda')

                # randomly choose an ensemble member
                n = np.random.randint(num_ensemble, size=1)[0]
                cfg["eval"]["ckpt_dir"] = os.path.join(dir_prefix,
                                                       str(n+1)+"/")
                model, _ = self._build_model(cfg)

                output = model(state_action)

                # learn residuals
                recover_res = \
                    self.norm.res_inv_normalize(output.cpu().numpy())
                target_res = \
                    self.norm.res_inv_normalize(target.cpu().numpy())
                recover_state = \
                    self.norm.inv_normalize(state_orig[None, None, :])
                recover_output = recover_res + recover_state
                recover_target = target_res + recover_state

                rollout_pred = \
                    self.norm.normalize(recover_output[:, None, :])

                loss = criterion(torch.Tensor(recover_output),
                                 torch.Tensor(recover_target))

                # output euler angles
                pred_angle = \
                    (R.from_matrix(recover_output[:, 6:].reshape(-1, 3, 3))
                      .as_euler('xyz', degrees=True))
                target_angle = \
                    (R.from_matrix(recover_target[:, 6:].reshape(-1, 3, 3))
                      .as_euler('xyz', degrees=True))

                losses.append(loss.item())
                preds.extend(np.hstack((recover_output, pred_angle)))
                targets.extend(np.hstack((recover_target, target_angle)))

        self.losses += np.sum(losses)
        return np.array(losses), np.array(preds), np.array(targets)
