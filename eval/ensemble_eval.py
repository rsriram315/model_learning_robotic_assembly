import os
from copy import deepcopy

from eval.mlp_eval import Evaluate


class EnsembleEvaluate:
    def __init__(self, cfg):
        self.cfg = deepcopy(cfg)

    def evaluate(self):
        num_ensemble = self.cfg["trainer"]["num_ensemble"]
        dir_prefix = self.cfg["trainer"]["ckpts_dir"]

        for n in range(num_ensemble):
            self.cfg["eval"]["ckpt_dir"] = os.path.join(dir_prefix,
                                                        str(n+1)+"/")
            eval = Evaluate(self.cfg)
            eval.evaluate()
