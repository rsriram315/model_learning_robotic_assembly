import setup  # noqa
import torch
import numpy as np
from .vis_mlp import Visualize


class ResVisualize(Visualize):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _evaluate(self, model, dataset):
        # get function handles of loss and metrics
        criterion = torch.nn.MSELoss()

        losses = []
        preds = []

        with torch.no_grad():
            for i in range(len(dataset)):
                state_action, target = dataset.__getitem__(i)

                state_action = state_action[np.newaxis, ...]
                target = target[np.newaxis, ...]
                state_action = torch.tensor(state_action).to('cuda')
                target = torch.tensor(target).to('cuda')

                output = model(state_action)

                loss = criterion(output, target)

                new_output = (output.cpu().numpy()[0, :] *
                              self.ds_stats["stat_4"] +
                              self.ds_stats["stat_3"] +
                              state_action.cpu().numpy()[0, :9])
                preds.append(new_output)
                losses.append(loss.item())
        return np.array(losses), np.array(preds)
