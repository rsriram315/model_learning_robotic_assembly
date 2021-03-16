from visualization import Visualize, EnsembleVisualize, MCDropoutVisualize
# from visualization.vis_features import FeaturesVisualize


def visualize(cfg):
    trainer_name = cfg["trainer"]["name"]
    model_name = cfg["model"]["name"]

    if trainer_name == "mlp":
        if model_name == "MLP":
            vis = Visualize(cfg)
        elif model_name == "MCDropout":
            vis = MCDropoutVisualize(cfg)
        # vis = FeaturesVisualize(cfg)
    elif trainer_name == "ensemble":
        vis = EnsembleVisualize(cfg)

    vis.visualize()
