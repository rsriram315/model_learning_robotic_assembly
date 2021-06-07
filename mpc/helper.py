import h5py
import numpy as np
from pathlib import Path


def get_goal_pos(data_dir_name="/home/paj7rng/amira_ML/data"):
    data_dir = Path(data_dir_name)
    recording_ls = list(data_dir.glob("*.h5"))

    # get goal pos only
    goal_pos = []
    for recording in recording_ls:
        with h5py.File(recording, 'r') as f:
            goal_pos.append(np.array(
                f['PandaStatePublisherarm_states']['tcp_pose_base'])[-1, :3])
    goal_pos = np.array(goal_pos)

    return np.mean(goal_pos, axis=0)
