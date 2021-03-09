from visualization import Visualize, EnsembleVisualize, ResVisualize
# from visualization.vis_features import FeaturesVisualize


def visualize(cfg):
    model_name = cfg["trainer"]["name"]

    if model_name == "mlp":
        if cfg["dataset"]["learn_residual"]:
            vis = ResVisualize(cfg)
        else:
            vis = Visualize(cfg)
            # vis = FeaturesVisualize(cfg)
    elif model_name == "ensemble":
        vis = EnsembleVisualize(cfg)

    vis.visualize()
