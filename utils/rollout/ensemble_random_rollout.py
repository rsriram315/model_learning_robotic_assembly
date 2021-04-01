import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from .mlp_rollout import Rollout


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True


class EnsembleRandomRollout(Rollout):
    def __init__(self, cfg, horizon):
        cfg["eval"]["ckpt_dir"] = "saved/ckpts/1/"
        super().__init__(cfg, horizon)
        self.horizon = horizon
        self.losses = 0

    def _evaluate(self, model, dataset):
        dir_prefix = "saved/ckpts/"

        rollout = self.horizon  # current rollout step
        rollout_pred = None

        # get function handles of loss and metrics
        criterion = torch.nn.MSELoss(reduction='sum')
        losses = []
        preds = []
        targets = []

        with torch.no_grad():
            for i in range(len(dataset)):
                state_action, target = dataset.__getitem__(i)

                if rollout < self.horizon:
                    state_action[:15] = rollout_pred
                    rollout += 1
                else:
                    rollout = 0

                state_action = torch.tensor(state_action[None, ...]).to('cuda')
                target = torch.tensor(target[None, ...]).to('cuda')

                n = np.random.randint(self.cfg["trainer"]["num_ensemble"],
                                      size=1)[0]
                self.cfg["eval"]["ckpt_dir"] = os.path.join(dir_prefix,
                                                            str(n+1)+"/")
                model, _ = self._build_model(self.cfg)

                output = model(state_action)
                rollout_pred = output.clone().cpu().numpy()

                loss = criterion(output, target)

                if self.learn_residual:
                    new_res = self.norm.res_inv_normalize(output.cpu().numpy())
                    target_res = self.norm.res_inv_normalize(target.cpu()
                                                                   .numpy())
                    new_state = self.norm.inv_normalize(
                                    state_action.cpu().numpy()[:15],
                                    is_state=True)
                    new_output = new_res + new_state
                    new_target = target_res + new_state
                else:
                    new_output = \
                        self.norm.inv_normalize(output.cpu().numpy(),
                                                is_state=True)
                    new_target = \
                        self.norm.inv_normalize(target.cpu().numpy(),
                                                is_state=True)

                # output euler angles
                pred_angle = \
                    (R.from_matrix(new_output[:, 6:].reshape(-1, 3, 3))
                      .as_euler('xyz', degrees=True))
                target_angle = \
                    (R.from_matrix(new_target[:, 6:].reshape(-1, 3, 3))
                      .as_euler('xyz', degrees=True))

                losses.append(loss.item())
                preds.extend(np.hstack((new_output, pred_angle)))
                targets.extend(np.hstack((new_target, target_angle)))

        return np.array(losses), np.array(preds), np.array(targets)