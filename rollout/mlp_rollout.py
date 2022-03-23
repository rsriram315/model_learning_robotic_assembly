import torch
import numpy as np
from copy import deepcopy
from pathlib import Path

from visualization.vis_mlp import MLPVisualize
from dataloaders.data_processor import Normalization, recover_rotation,\
                                       add_euler_angle


class MLPRollout(MLPVisualize):
    def __init__(self, cfg, horizon):
        super().__init__(cfg, vis_dir=f"saved/visualizations_{horizon}")
        self.horizon = horizon
        self.losses = []

    def visualize(self):
        cfg = deepcopy(self.cfg)
        model, cfg = self._build_model(cfg)
        self.norm = Normalization(cfg["dataset"]["stats"])

        for fname in self.demo_fnames:
            if fname in self.train_demo_fnames:
                suffix_fname = Path('train') / Path(fname).stem
            elif fname in self.test_demo_fnames:
                suffix_fname = Path('test') / Path(fname).stem

            # read dataset
            dataset = self._read_single_demo(cfg["dataset"], [fname])
            time = dataset.sample_time
            losses_per_demo, preds_per_demo, state_per_demo =\
                self._evaluate(model, dataset)
            preds_per_demo, target_per_demo = \
                self._recover_data(preds_per_demo, state_per_demo)

            if self.vis_cfg["loss"]:
                loss_fname = self.vis_dir / "loss" / suffix_fname
                print(losses_per_demo.shape)
                print(target_per_demo.shape)
                print(len(time[2:]))
                self._vis_loss(losses_per_demo, time[2:], loss_fname)

            if self.vis_cfg["axis"]:
                axis_fname = self.vis_dir / "axis" / suffix_fname
                self._vis_axis(preds_per_demo,
                               target_per_demo,
                               time[2:],
                               axis_fname)

            if self.vis_cfg["trajectory"]:
                traj_fname = self.vis_dir / "trajectory" / suffix_fname
                self._vis_trajectory(preds_per_demo[:, :3],
                                     target_per_demo[:, :3],
                                     traj_fname)

            print(f"... Generated visualization for {fname}\n")

    def _evaluate(self, model, dataset):
        rollout = self.horizon  # current rollout step
        ro_pred = None  # rollout_predictions

        # get function handles of loss and metrics
        gt_states = []
        ro_states = []
        preds = []

        criterion = torch.nn.MSELoss(reduction='none')
        losses = []

        with torch.no_grad():
            for i in range(len(dataset) - 1):
                state_action, target = dataset.__getitem__(i)
                gt_states.append(np.copy(state_action[None, :12]))

                if rollout < self.horizon:
                    state_action[:12] = np.copy(ro_pred)
                    rollout += 1
                else:
                    rollout = 0
                ro_states.append(np.copy(state_action[None, :12]))

                # batch_size x 12
                state_action = torch.tensor(state_action[None, ...]).to(self.device)
                target = torch.tensor(target[None, ...]).to(self.device)
                output = model(state_action)  # predicted difference

                # get rollout prediction in the correct scale
                # and this is the new state for next prediction
                ro_pred = output.cpu().numpy()
                ro_pred = self.norm.inv_normalize(ro_pred[:, None, :],
                                                  is_res=True)
                ro_pred[:, :3] += self.norm.inv_normalize(ro_states[-1]
                                                          [:, None, :])[:, :3]
                ro_pred = recover_rotation(ro_pred, ro_states[-1])
                ro_pred = self.norm.normalize(ro_pred[:, None, :])

                # get ground truth next state
                next_state_action, _ = dataset.__getitem__(i+1)

                # loss is set to the difference between
                # predicted current state (normalized) and
                # the ground truth next states
                loss = criterion(torch.tensor(ro_pred),
                                 torch.tensor(next_state_action[None, :12]))
                loss = torch.mean(loss, axis=1)

                losses.extend(loss.cpu().numpy())
                preds.extend(ro_pred)

        self.losses.extend(losses)
        return (np.array(losses), np.array(preds), np.array(gt_states))

    def _recover_data(self, pred, state):
        recover_output = np.copy(pred)
        recover_output[:, :12] = self.norm.inv_normalize(pred[:, None, :12])
        recover_output = add_euler_angle(recover_output)

        recover_target = self.norm.inv_normalize(state)
        recover_target = add_euler_angle(recover_target)
        return recover_output, recover_target
