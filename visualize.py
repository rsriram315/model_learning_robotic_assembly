from visualization import Visualize, EnsembleVisualize


def visualize(cfg):
    model_name = cfg["trainer"]["name"]

    if model_name == "mlp":
        vis = Visualize(cfg)
    elif model_name == "ensemble":
        vis = EnsembleVisualize(cfg)

    vis.visualize()
