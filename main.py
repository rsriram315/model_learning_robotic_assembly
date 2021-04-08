import setup  # noqa
import argparse
from train import train
from evaluate import evaluate
from visualize import visualize
from utils import read_json
from rollout import rollout  # noqa


def main(cfg_path):
    # setup dataloader instances
    cfg = read_json(cfg_path)

    if cfg["train"]:
        train(cfg)
    if cfg["evaluate"]:
        evaluate(cfg)
    if cfg["visualize"]:
        visualize(cfg)
    if cfg["rollout"]:
        rollout(cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('-c',
                        '--config',
                        default=None,
                        type=str,
                        help='config file path (default: None)')
    args = parser.parse_args()
    main(args.config)
