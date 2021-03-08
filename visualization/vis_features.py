import matplotlib.pyplot as plt
import numpy as np
from .vis_mlp import Visualize


class FeaturesVisualize(Visualize):
    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__(cfg)

    def _vis_axis(self, pred, state, time, fname):
        fig, axs = plt.subplots(1, 3, figsize=(20, 7), sharex='all')

        pred = np.array(pred)

        # the predicted states should start with the prediction for t=1 not t=0
        # for t in range(1, len(time), subsample):
        size = 1
        features = ['rot']
        axis = ['x', 'y', 'z']

        for _, feature in enumerate(features):
            for c, ax in enumerate(axis):
                idx = c
                axs[c].scatter(time[1:], state[1:, idx + 3], s=size,
                               c='tab:blue', label="ground truth")
                axs[c].scatter(time[1:], pred[:-1, idx], s=size,
                               c='tab:orange', label="predictions")
                axs[c].set_title(f'{feature} {ax} axis')
                axs[c].set_ylabel('coordinate')
                axs[c].legend()
                axs[c].set_xlabel('time')
        plt.savefig(fname)
        plt.close(fig)
