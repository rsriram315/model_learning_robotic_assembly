import matplotlib.pyplot as plt
from utils.rollout import Rollout, MCRollout, EnsembleRollout,\
                          EnsembleRandomRollout


def rollout(cfg):
    losses = []
    horizons = [5, 10, 15, 20]

    for h in horizons:
        if cfg["name"] == "mlp":
            vis = Rollout(cfg, horizon=h)
        elif cfg["name"] == "mc_dropout":
            vis = MCRollout(cfg, horizon=h)
        elif cfg["name"] == "ensemble":
            # vis = EnsembleRollout(cfg, horizon=h)
            vis = EnsembleRandomRollout(cfg, horizon=h)
        vis.visualize()
        losses.append(vis.losses)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.set_title("rollout horizon and loss")
    ax.plot(horizons, losses, 'o-')
    ax.set_xlabel("horizon")
    ax.set_ylabel("loss")

    plt.tight_layout()
    plt.savefig("./saved/horizon_loss", dpi=200)
    plt.close(fig)
