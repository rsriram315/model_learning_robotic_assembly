import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils.visualization import MLPVisualize


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
                gt_states.append(np.copy(state_action[:15]))
                targets.append(np.copy(target))

                if rollout < self.horizon:
                    state_action[:15] = np.copy(ro_pred)
                    rollout += 1
                else:
                    rollout = 0
                ro_states.append(np.copy(state_action[:15]))

                # batch_size x 15
                state_action = torch.tensor(state_action[None, ...]).to('cuda')
                target = torch.tensor(target[None, ...]).to('cuda')
                output = model(state_action)

                # get rollout prediction in the correct scale
                ro_pred = output.cpu().numpy()
                ro_pred = self.norm.inv_normalize(ro_pred[:, None, :],
                                                  is_res=True)
                ro_pred += self.norm.inv_normalize(ro_states[-1]
                                                   [None, None, :])
                ro_pred = self.norm.normalize(ro_pred[:, None, :])

                loss = criterion(torch.tensor(ro_pred),
                                 torch.tensor(gt_states[-1][None, ...]))
                loss = torch.sum(loss, dim=1)
                # ro_pred = output.clone().cpu().numpy()
                # loss = criterion(torch.tensor(ro_pred),
                #                  torch.tensor(gt_states[-1][None, ...]))
                # loss = torch.sum(loss, dim=1)

                losses.extend(loss.cpu().numpy())
                preds.extend(output.cpu().numpy())

        self.losses.extend(losses)
        return (np.array(losses), np.array(preds),
                (np.array(gt_states), np.array(ro_states)),
                np.array(targets))

    def _recover_data(self, pred, state, target):
        gt_state, rollout_state = state

        curr_gt_state = self.norm.inv_normalize(gt_state[:, None, :])
        curr_rollout_state = self.norm.inv_normalize(rollout_state[:, None, :])

        recover_target = self.norm.inv_normalize(target[:, None, :],
                                                 is_res=True)
        recover_output = self.norm.inv_normalize(pred[:, None, :],
                                                 is_res=True)
        recover_target += curr_gt_state
        recover_output += curr_rollout_state

        # output euler angles
        pred_angle = (R.from_matrix(recover_output[:, 6:].reshape(-1, 3, 3))
                       .as_euler('xyz', degrees=True))
        target_angle = (R.from_matrix(recover_target[:, 6:].reshape(-1, 3, 3))
                         .as_euler('xyz', degrees=True))

        recover_output = np.hstack((recover_output, pred_angle))
        recover_target = np.hstack((recover_target, target_angle))
        return recover_output, recover_target

    # for learning absolute values
    # def _recover_data(self, pred, state, target):
    #     recover_target = self.norm.inv_normalize(target[:, None, :])
    #     recover_output = self.norm.inv_normalize(pred[:, None, :])

    #     # output euler angles
    #     pred_angle = (R.from_matrix(recover_output[:, 6:].reshape(-1, 3, 3))
    #                    .as_euler('xyz', degrees=True))
    #     target_angle = (R.from_matrix(recover_target[:, 6:]
    #                      .reshape(-1, 3, 3))
    #                      .as_euler('xyz', degrees=True))

    #     recover_output = np.hstack((recover_output, pred_angle))
    #     recover_target = np.hstack((recover_target, target_angle))
    #     return recover_output, recover_target
