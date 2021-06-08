import h5py
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R


def get_goal(data_dir_name="/home/paj7rng/amira_ML/data"):
    data_dir = Path(data_dir_name)
    recording_ls = list(data_dir.glob("*.h5"))

    # get goal pos only
    goal_pos_ls = []
    goal_orn_quat_ls = []
    for recording in recording_ls:
        with h5py.File(recording, 'r') as f:
            goal_pos_ls.append(np.array(
                f['PandaStatePublisherarm_states']['tcp_pose_base'])[-1, :3])
            goal_orn_quat_ls.append(np.array(
                f['PandaStatePublisherarm_states']['tcp_pose_base'])[-1, 3:])

    goal_pos = np.mean(goal_pos_ls, axis=0)
    goal_orn = R.from_quat(goal_orn_quat_ls).mean().as_matrix()

    return goal_pos, goal_orn
