import torch
import numpy as np
from utils.visualization import MLPVisualize
from dataloaders import recover_rotation, add_euler_angle


class MLPRollout(MLPVisualize):
    def __init__(self, cfg, horizon):
        super().__init__(cfg, vis_dir=f"saved/visualizations_{horizon}")
        self.horizon = horizon
        self.losses = []

    def _evaluate(self, model, dataset):
        rollout = self.horizon  # current rollout step
        ro_pred = None  # rollout_predictions

        # get function handles of loss and metrics
        criterion = torch.nn.MSELoss(reduction='none')
        losses = []
        preds = []
        gt_states = []
        ro_states = []
        targets = []

        with torch.no_grad():
            for i in range(len(dataset)):
                state_action, target = dataset.__getitem__(i)
                gt_states.append(np.copy(state_action[None, :15]))
                targets.append(np.copy(target[None, :]))

                if rollout < self.horizon:
                    state_action[:15] = np.copy(ro_pred)
                    rollout += 1
                else:
                    rollout = 0
                ro_states.append(np.copy(state_action[None, :15]))

                # batch_size x 15
                state_action = torch.tensor(state_action[None, ...]).to('cuda')
                target = torch.tensor(target[None, ...]).to('cuda')
                output = model(state_action)

                # get rollout prediction in the correct scale
                ro_pred = output.cpu().numpy()
                ro_pred = self.norm.inv_normalize(ro_pred[:, None, :],
                                                  is_res=True)
                ro_pred[:, :6] += self.norm.inv_normalize(ro_states[-1]
                                                          [:, None, :])[:, :6]
                ro_pred = self.norm.normalize(ro_pred[:, None, :])
                ro_pred = recover_rotation(ro_pred, ro_states[-1])

                loss = criterion(torch.tensor(ro_pred),
                                 torch.tensor(gt_states[-1]))
                loss = torch.mean(loss, axis=1)

                losses.extend(loss.cpu().numpy())
                preds.extend(ro_pred)

        self.losses.extend(losses)
        return (np.array(losses), np.array(preds), np.array(gt_states),
                np.array(targets))

    def _recover_data(self, pred, gt_state, target):
        recover_output = np.copy(pred)
        recover_output[:, :15] = self.norm.inv_normalize(pred[:, None, :15])
        recover_output = add_euler_angle(recover_output)

        curr_gt_state = self.norm.inv_normalize(gt_state)
        recover_target = self.norm.inv_normalize(target, is_res=True)
        recover_target[:, :6] += curr_gt_state[:, :6]

        # recover rotation matrix and add euler angles
        recover_target = recover_rotation(recover_target, curr_gt_state)
        recover_target = add_euler_angle(recover_target)
        return recover_output, recover_target
