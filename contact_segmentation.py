import os
import numpy as np
from pathlib import Path
from utils.visualization import read_demo, novelty_energy,\
                                contact_time, vis_contact_seg


if __name__ == '__main__':
    # create dirs
    vis_dir = Path("contact_seg")
    Path(vis_dir).mkdir(parents=True, exist_ok=True)

    demos_dir = Path("data/")
    demos_fnames = list(demos_dir.glob("*.h5"))

    subsample = 2

    for fname in demos_fnames:
        dataset = read_demo(os.fspath(fname.name))

        force_xyz = np.array(dataset.states_force)
        time = np.array(dataset.sample_time)

        novelty, energy, magnitude = novelty_energy(force_xyz, time)

        force_half = force_xyz[::subsample]
        time_half = time[::subsample]
        novelty_half, energy_half, _ = novelty_energy(force_half, time_half)
        start, end = contact_time(novelty_half, energy_half, time_half)
        # start = [0]
        # end = [-1]

        vis_fname = vis_dir / fname.stem

        vis_contact_seg(novelty, energy, magnitude,
                        time, start, end, vis_fname)
        print(f"Generated visualization for {vis_fname}")
