from visualization import MLPVisualize, EnsembleVisualize,\
                                MCDropoutVisualize


def visualize(cfg):
    trainer_name = cfg["trainer"]["name"]
    model_name = cfg["model"]["name"]

    if trainer_name == "mlp":
        if model_name == "MLP":
            vis = MLPVisualize(cfg)
        elif model_name == "MCDropout":
            vis = MCDropoutVisualize(cfg)
    elif trainer_name == "ensemble":
        vis = EnsembleVisualize(cfg)

    vis.visualize()
