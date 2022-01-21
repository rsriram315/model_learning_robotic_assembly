import h5py
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R


def get_goal(data_dir_name=None):
    data_dir = Path(data_dir_name)
    recording_ls = list(data_dir.glob("*.h5"))

    # get goal position and orientation from data
    goal_pos_ls = []
    goal_orn_quat_ls = []
    for recording in recording_ls:
        with h5py.File(recording, 'r') as f:
            goal_pos_ls.append(np.array(
                f['PandaStatePublisherarm_states']['tcp_pose_base'])[-1, :3])
            goal_orn_quat_ls.append(np.array(
                f['PandaStatePublisherarm_states']['tcp_pose_base'])[-1, 3:])

    # manually change values for prototyping
    goal_pos = np.array([0.269, -0.412,  0.1825]) # for easyinsertion 0.400, 0.376, 0.285 # hard insert 0.265, -0.411,  0.183
    # goal_orn_quat_ls = np.array([0.973, -0.226, -0.041, 0.007]) # for easy insertion task 1, 0.25, 0.000, 0 # hard insert 0.972, -0.227, -0.045, -0.011
    # goal_orn_quat_ls = goal_orn_quat_ls / np.linalg.norm(goal_orn_quat_ls)
    # goal_orn = R.from_quat(goal_orn_quat_ls).as_matrix()
    
    # get goal from demo data
    # goal_pos = np.mean(goal_pos_ls, axis=0)
    goal_orn = R.from_quat(goal_orn_quat_ls).mean().as_matrix()

    return goal_pos, goal_orn


def calculate_costs(resulting_states_ls, goal, cost_fn):
    """
    Rank various predicted trajectories (by cost)

    Args:
        resulting_states_ls:
            predicted trajectories [horizon, N, state_size]
        actions:
            the actions that were "executed" in order to achieve the predicted
            trajectories [N, h, action_size]
        reward_func:
            calculates the rewards associated with each state transition in
            the predicted trajectories

    Returns:
        cost_for_ranking: cost associated with each candidate action sequence
        [N,]
    """

    ###########################################################
    # calculate costs associated with each predicted trajectory
    ###########################################################
    #init vars for calculating costs
    horizon, num_sample_seq, _ = resulting_states_ls.shape
 
    # accumulate cost over each timestep
    costs = []
    for traj in range(num_sample_seq):
        cost = 0
        for h in range(horizon-1, -1, -1):
            cost = cost_fn(resulting_states_ls[h, traj, :], goal) + (0.9 * cost)
        costs.append(cost)

    return np.array(costs)
