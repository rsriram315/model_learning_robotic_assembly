import numpy as np
import matplotlib.pyplot as plt

from rollout.mlp_rollout import MLPRollout
from rollout.mc_rollout import MCRollout
from rollout.ensemble_rollout import EnsembleRollout
# EnsembleRandomRollout


def rollout(cfg):
    losses = []
    horizons = [5, 10, 15, 20]
    # horizons = [20]

    for h in horizons:
        if cfg["name"] == "mlp":
            vis = MLPRollout(cfg, horizon=h)
        elif cfg["name"] == "mc_dropout":
            vis = MCRollout(cfg, horizon=h)
        elif cfg["name"] == "ensemble":
            vis = EnsembleRollout(cfg, horizon=h)
            # vis = EnsembleRandomRollout(cfg, horizon=h)
        vis.visualize()
        losses.append(np.mean(vis.losses))

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.set_title("rollout horizon and loss")
    ax.plot(horizons, losses, 'o-')
    ax.set_xlabel("horizon")
    ax.set_ylabel("loss")

    plt.tight_layout()
    plt.savefig("./saved/horizon_loss.eps", dpi=1200, format='eps')
    plt.close(fig)
