import torch

from eval.mlp_eval import Evaluate
from eval.ensemble_eval import EnsembleEvaluate


# fix random seeds for reproducibility
# SEED = 123
# torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
# np.random.seed(SEED)


def evaluate(cfg):
    trainer_name = cfg["trainer"]["name"]

    if trainer_name == "mlp":
        eval = Evaluate(cfg)
    elif trainer_name == "ensemble":
        eval = EnsembleEvaluate(cfg)

    print("... Evaluating trained model\n")
    eval.evaluate()
