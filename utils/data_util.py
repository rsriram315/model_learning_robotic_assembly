import numpy as np
import h5py

# gripper distance smaller than 0.013 means 'grasp', otherwise it's release
GRIPPER_CLOSED_THRESHOLD = 0.013


def grasp_time(states, time_stamp):
    """
    detect when the gripper grasp and release the object
    only one start time and one stop time are assumed in this implemetation

    Params:
        states: state of the gripper joint, q
        time_stamp: time stamp of gripper joint states
    Return:
        start_time: the first recorded time when the gripper grasp the object
        stop_time: the last recorded time when the gripper grasp the object
    """
    mask_thres = [s > GRIPPER_CLOSED_THRESHOLD for s in states[:, 0]]
    mask_t = np.convolve(list(map(int, mask_thres)), [1, -1], 'valid')
    start_idx = np.argmin(mask_t) + 1
    stop_idx = np.argmax(mask_t)
    start_time = time_stamp[start_idx]
    stop_time = time_stamp[stop_idx]
    return start_time, stop_time


def read_one_demo(data_path, states, actions, time="time_stamp"):
    """
    read data from h5 file
    should I discard the time when the arm is not grasping the object?

    Params:
        data_path: data file path
        sample_freq: sampling frequency
        states: specified states for current task
        actions: specified actions for current task
        time: name of time stamp in the data
    Return:
        tuple of states (state, time), actions (action, time).
        state and action are (position, orientation) here.
    """
    with h5py.File(data_path, 'r') as f:
        # determine when the arm grasp and release object
        gripper_t = np.array(f['franka_gripperjoint_states'][time])
        gripper_s = np.array(f['franka_gripperjoint_states']['q'])
        grasp_start_t, grasp_stop_t = grasp_time(gripper_s, gripper_t)

        data_time = [{}, {}]  # [[state, time], [action, time]]
        for n, i in enumerate([states, actions]):
            # clip time to avoid negative value
            t = np.clip(np.asarray(f[i["name"]][time]), 0, None)

            start_idx = np.argmin(abs(t - grasp_start_t))
            stop_idx = np.argmin(abs(t - grasp_stop_t))
            # discard the data when the arm not grasping the object
            d = np.array(f[i["name"]][i["attrs"]])[start_idx:stop_idx]
            t = t[start_idx:stop_idx]

            data_time[n]["data"] = d
            data_time[n]["time"] = t
    return data_time


def pair_state_action(sample_freq, states_time, actions_time):
    """
    form state action pair, state has more dense data than action.
    choosing the action which is closest to the state.
    """
    state_action_idx = []
    # actions_time["time"], states_time["time"]

    start_time = max(min(states_time["time"]), min(actions_time["time"]))
    end_time = max(max(states_time["time"]), max(actions_time["time"]))

    # for i in [states_time, actions_time]:
    #     mask = list(map(int, (i["time"] - start_time) < 0))
    #     start_idx = sum(mask)
    #     i["data"] = i["data"][start_idx:]
    #     i["time"] = i["time"][start_idx:]

    sample_time = np.arange(start_time, end_time, 1.0/int(sample_freq))

    for t in sample_time:
        state_action_idx.append(
            zero_order_interpolation(t, states_time["time"],
                                     actions_time["time"]))
    return state_action_idx


def zero_order_interpolation(time_stamp, states_t, actions_t):
    """
    not exactly a zero order, we choose the nearest neighbor data
    BEFORE the sampling time stamp.
    """
    state_mask = list(map(int, (states_t - time_stamp) <= 0))
    action_mask = list(map(int, (actions_t - time_stamp) <= 0))
    state_idx = sum(state_mask) - 1
    action_idx = sum(action_mask) - 1

    return state_idx, action_idx
