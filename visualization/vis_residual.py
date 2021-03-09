import setup  # noqa
import torch
import numpy as np
from .vis_mlp import Visualize


class ResVisualize(Visualize):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _evaluate(self, model, dataset):
        losses = []
        preds = []
        criterion = torch.nn.MSELoss()

        with torch.no_grad():
            for i in range(len(dataset)):
                state_action, target = dataset.__getitem__(i)

                # state_action = state_action[np.newaxis, ...]
                # target = target[np.newaxis, ...]
                state_action = torch.tensor(state_action).to('cuda')
                target = torch.tensor(target).to('cuda')

                output = model(state_action)

                loss = criterion(output, target)

                new_res = self.norm.residual_inv_normalize(
                               output.cpu().numpy())
                new_state = self.norm.inverse_normalize(
                                state_action.cpu().numpy()[:9],
                                is_target=True)[0]

                preds.append(new_res + new_state)
                losses.append(loss.item())
        return np.array(losses), np.array(preds)
