import torch
import numpy as np
from copy import deepcopy
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from dataloaders import Normalization, Standardization
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

        if self.cfg["dataset"]["preprocess"]["normalize"]:
            self.norm = Normalization(ds_stats)
        elif self.cfg["dataset"]["preprocess"]["standardize"]:
            self.norm = Standardization(ds_stats)

        loss_mean = []
        loss_std = []
        pred_mean = []
        pred_std = []
        targets = []
        time = []

        for fname in self.demo_fnames:
            fname = Path(fname)
            dataset = self._read_single_demo(deepcopy(self.cfg["dataset"]),
                                             [fname.name], ds_stats)
            time.append(dataset.sample_time)

            losses, preds_mean, preds_std, target = self._evaluate(model,
                                                                   dataset)

            loss_mean.append(losses)
            loss_std.append(np.zeros_like(losses))
            pred_mean.append(preds_mean)
            pred_std.append(preds_std)
            targets.append(target)

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
                s_a, t = dataset.__getitem__(i)

                if rollout < self.horizon:
                    s_a[:15] = rollout_pred
                    rollout += 1
                else:
                    rollout = 0

                state_action = np.vstack([s_a for _ in range(num_mc)])
                state_action = torch.tensor(state_action).to('cuda')
                target = np.vstack([t for _ in range(num_mc)])
                target = torch.tensor(target).to('cuda')

                output = model(state_action)
                loss = criterion(output, target)
                rollout_pred = torch.mean(output, dim=0).cpu().numpy()

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

                losses.append(loss.item() / num_mc)
                preds_mean.append(np.hstack((np.mean(new_output, axis=0),
                                            np.mean(pred_angle, axis=0))))
                preds_std.append(np.hstack((np.std(new_output, axis=0),
                                           np.std(pred_angle, axis=0))))
                targets.append(np.hstack((new_target, target_angle))[0])

        return (np.array(losses), np.array(preds_mean), np.array(preds_std),
                np.array(targets))
