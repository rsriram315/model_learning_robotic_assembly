import setup  # noqa
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from dataloaders import DemoDataset
from utils import read_json


def main():
    # create dirs
    vis_dir = Path("visualization/vis_contact_seg")
    Path(vis_dir).mkdir(parents=True, exist_ok=True)

    demos_dir = Path("data/")
    demos_fnames = list(demos_dir.glob("*.h5"))

    subsample = 2

    for fname in demos_fnames:
        dataset = read_demo([fname.name])

        force_xyz = dataset.all_states_force
        time = dataset.all_states_time

        novelty, energy, magnitude = novelty_energy(force_xyz, time)

        force_half = force_xyz[::subsample]
        time_half = time[::subsample]
        novelty_half, energy_half, _ = novelty_energy(force_half, time_half)
        start, end = contact_time(novelty_half, energy_half, time_half)

        vis_fname = vis_dir / fname.stem

        vis_contact_seg(novelty, energy, magnitude,
                        time, start, end, vis_fname)
        print(f"Generated visualization for {vis_fname}")


def read_demo(fname):
    # setup dataloader instances
    cfg = read_json("config.json")
    ds_cfg = cfg["dataset"]
    ds_cfg["params"]["fnames"] = fname
    ds_cfg["params"]["contact_only"] = False
    # dataset_cfg["params"]["process"]["normalize"] = false

    dataset = DemoDataset(**ds_cfg["params"])

    return dataset


def vis_contact_seg(novelty, energy, magnitude,
                    time, start, end, fname):
    fig, axs = plt.subplots(2, 1, figsize=(25, 10), sharex='all')

    axs[0].plot(time, novelty, label="low-pass novelty")
    axs[0].set_ylabel("novelty")
    axs[0].legend()
    for s in start:
        axs[0].axvline(x=time[s], color='tab:red', linestyle='--')
    for e in end:
        axs[0].axvline(x=time[e], color='tab:green', linestyle='--')

    axs[1].plot(time, magnitude, label="magnitude")
    axs[1].plot(time, energy, label="low-pass magnitude")
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


def contact_time(novelty, energy, time,
                 novelty_thres=0.15,
                 energy_thres=0.15,
                 subsample=2):
    """
    detecting when the robot contacts the environment

    Params:
        force_xyz: force of x y z axis
    Return:
        start and end index in time
    """

    # start point should be small, and have large positive novelty
    start_mask = (novelty >= novelty_thres) & (energy <= energy_thres)
    start_candidate = np.where(start_mask)[0]

    # end point should also be small, and have large negative novelty
    end_mask = (novelty <= -novelty_thres) & (energy <= energy_thres)
    end_candidate = np.where(end_mask)[0]

    # if the last energy is not small
    # it could be also an end point (sudden stop of recording)
    if energy[-1] >= energy_thres:
        end_candidate = np.append(end_candidate, time.size - 1)

    # suppress noisy detected boundaries
    start = find_start_bounds(start_candidate)
    end = find_end_bounds(end_candidate)
    start, end = match_start_end(start, end, subsample)

    return start, end


def novelty_energy(force_xyz, time):
    # use start phase force as baseline
    force_xyz = calibrate_force(force_xyz)

    # use the force magnitude sum
    force = np.linalg.norm(force_xyz, axis=1)

    magnitude = force.copy()
    magnitude /= np.amax(magnitude)  # normalized

    # low-pass filter for forces, otherwise would be too noisy
    force_lp = gaussian_filter1d(force, sigma=8)

    # energy = force_lp**2  # power 2 to suppress small values
    energy = force_lp
    energy /= np.amax(energy)  # normalize energy

    # discrete dervative
    energy_diff = np.diff(energy)
    time_diff = np.diff(time)  # time difference

    novelty = energy_diff / time_diff
    novelty = np.append(novelty, 0.0)
    novelty /= np.amax(abs(novelty))  # normalize to [-1, 1]
    return novelty, energy, magnitude


def calibrate_force(force_xyz, start_period=10):
    """
    use start phase force as baseline for contact detection

    force_xyz has shape (num_data, 3)
    """
    offset_x = np.mean(force_xyz[:start_period, 0])
    offset_y = np.mean(force_xyz[:start_period, 1])
    offset_z = np.mean(force_xyz[:start_period, 2])

    force_xyz[:, 0] -= offset_x
    force_xyz[:, 1] -= offset_y
    force_xyz[:, 2] -= offset_z

    return force_xyz


def find_start_bounds(candidate, tolerant=10):
    """
    find start boundaries, keep the first found bound
    """
    bounds = [candidate[0]]
    bound_prev = bounds[0]

    for idx in range(candidate.size):
        bound_new = candidate[idx]

        if bound_new - bound_prev >= tolerant:
            bounds.append(bound_new)

        bound_prev = bound_new
    return np.array(bounds)


def find_end_bounds(candidate, tolerant=10):
    """
    find end boundary, keep the last fined bound
    """
    bounds = [candidate[0]]

    for idx in range(candidate.size):
        bound_new = candidate[idx]

        if bound_new - bounds[-1] >= tolerant:
            bounds.append(bound_new)
        else:
            bounds[-1] = bound_new
    return np.array(bounds)


def match_start_end(start, end, subsample):
    """
    Assume only one contacting phase exits and it is continuous
    match the first found start and the last found end
    """
    return [start[0] * subsample], [end[-1] * subsample]


if __name__ == "__main__":
    main()
