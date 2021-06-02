import setup  # noqa
import argparse

from utils import read_json
from trainer import train
from eval import evaluate
from mpc import mpc_controller
from visualization import visualize
from visualization.vis_contact_force import vis_contact_force
from rollout import rollout


def main(cfg_path):
    # setup dataloader instances
    cfg = read_json(cfg_path)

    # visualize the segmentated contact force
    if cfg["visualization"]["contact"]:
        vis_contact_force()

    if cfg["train"]:
        train(cfg)
    if cfg["evaluate"]:
        evaluate(cfg)
    if cfg["visualize"]:
        visualize(cfg)
    if cfg["rollout"]:
        rollout(cfg)
    if cfg["mpc"]:
        mpc_controller(cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('-c',
                        '--config',
                        default=None,
                        type=str,
                        help='config file path (default: None)')
    args = parser.parse_args()
    main(args.config)
