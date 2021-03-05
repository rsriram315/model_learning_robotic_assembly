import torch
import numpy as np
from eval import Evaluate, EnsembleEvaluate


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
np.random.seed(SEED)


def evaluate(cfg):
    model_name = cfg["trainer"]["name"]

    if model_name == "mlp":
        eval = Evaluate(cfg)
    elif model_name == "ensemble":
        eval = EnsembleEvaluate(cfg)

    eval.evaluate()
