import numpy as np


# gripper distance smaller than 0.013 means 'grasp', otherwise it's release
GRIPPER_CLOSED_THRESHOLD = 0.013


def grasp_time(states, time_stamp):
    state = np.copy(states)
    state = np.around(state, 3)
    indices = np.where(state[:, 0] == GRIPPER_CLOSED_THRESHOLD)[0]
    start_time = time_stamp[indices[0]]
    stop_time = time_stamp[indices[-1]]
    return start_time, stop_time
