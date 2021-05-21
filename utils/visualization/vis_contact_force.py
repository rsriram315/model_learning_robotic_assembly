import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from utils import read_json
from dataloaders import DemoDataset, SegmentContact


def read_demo(fname):
    # setup dataloader instances
    cfg = read_json("configs/mlp.json")
    cfg["dataset"]["fnames"] = [fname]
    cfg["dataset"]["contact_only"] = False

    dataset = DemoDataset(cfg["dataset"])
    return dataset


def plot_seg(novelty, energy, time, start, end, fname):
    fig, axs = plt.subplots(2, 1, figsize=(25, 10), sharex='all')

    axs[0].plot(time, novelty, label="low-pass novelty")
    axs[0].set_ylabel("novelty")
    axs[0].legend()
    for s in start:
        axs[0].axvline(x=time[s], color='tab:red', linestyle='--')
    for e in end:
        axs[0].axvline(x=time[e], color='tab:green', linestyle='--')

    # axs[1].plot(time, magnitude, label="magnitude")
    axs[1].plot(time, energy, label="low-pass energy")
    axs[1].set_ylabel("energy")
    axs[1].legend()
    for s in start:
        axs[1].axvline(x=time[s], color='tab:red', linestyle='--')
    for e in end:
        axs[1].axvline(x=time[e], color='tab:green', linestyle='--')

    for s in start:
        axs[1].axvline(x=time[s], color='tab:red', linestyle='--')
    for e in end:
        axs[1].axvline(x=time[e], color='tab:green', linestyle='--')

    plt.savefig(fname)
    plt.close(fig)


def vis_contact_force(vis_dir="saved/contact_seg",
                      subsample=2,
                      energy_thres=0.3,
                      novelty_thres=0.15):
    # create dirs
    vis_dir = Path(vis_dir)
    Path(vis_dir).mkdir(parents=True, exist_ok=True)

    demos_dir = Path("data/")
    demos_fnames = list(demos_dir.glob("*.h5"))

    seg_contact = SegmentContact()

    print("... Visualizing Contact Force\n")

    for fname in demos_fnames:
        dataset = read_demo(os.fspath(fname.name))
        print(f"Segmenting contact force for {fname.name}")

        force_xyz = np.array(dataset.states_force)
        time = np.array(dataset.sample_time)

        novelty, energy = \
            seg_contact._novelty_energy(force_xyz, time)
        start, end = seg_contact.contact_time(force_xyz, time, subsample,
                                              energy_thres, novelty_thres)

        vis_fname = vis_dir / fname.stem

        plot_seg(novelty, energy, time, start, end, vis_fname)
        print(f"Generated visualization for {fname.name}\n")
