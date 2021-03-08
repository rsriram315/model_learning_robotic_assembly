from visualization.vis_features import FeaturesVisualize
from visualization import Visualize, EnsembleVisualize, ResVisualize,\
                          FeaturesVisualize


def visualize(cfg):
    model_name = cfg["trainer"]["name"]

    if model_name == "mlp":
        vis = Visualize(cfg)
        # vis = ResVisualize(cfg)
        # vis = FeaturesVisualize(cfg)
    elif model_name == "ensemble":
        vis = EnsembleVisualize(cfg)

    vis.visualize()
