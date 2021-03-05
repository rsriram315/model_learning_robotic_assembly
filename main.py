import setup  # noqa
import argparse
from copy import deepcopy
from train import train
from evaluate import evaluate
from visualize import visualize
from utils import read_json


def main(cfg_path):
    # setup dataloader instances
    cfg = read_json(cfg_path)

    if cfg["train"]:
        train(deepcopy(cfg))
    if cfg["evaluate"]:
        evaluate(deepcopy(cfg))
    if cfg["visualize"]:
        visualize(deepcopy(cfg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('-c',
                        '--config',
                        default=None,
                        type=str,
                        help='config file path (default: None)')
    args = parser.parse_args()
    main(args.config)
