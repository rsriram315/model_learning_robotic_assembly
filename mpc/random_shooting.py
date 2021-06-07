# flake8: noqa
import numpy as np
from mpc.helper import get_goal_pos


class RandomShooting(object):
    """
    Generate multiple random action rollouts, and select the best one
    """
    def __init__(self, env, dyn_model, cost_fn, rand_policy, params):
        self.horizon = params.horizon
        self.N = params.num_sample_seq  # number of random action sequences
        self.dyn_model = dyn_model
        self.cost_fn = cost_fn
        self.rand_policy = rand_policy
        self.env = env

        # TODO:deepcopy will generate error, don't know WHY
        # self.env = deepcopy(env)

        # TODO position range or even also rotation range
        self.random_sampling_params = {'sample_rot': params.sample_rot,
                                       'angle_min': params.rand_policy_angle_min,
                                       'angle_max': params.rand_policy_angle_max,
                                       'hold_action': params.rand_policy_hold_action}

    def get_action(self, curr_state):
        """
        Select optimal action

        Agrs:
            curr_state_K:
                current "state" as known by the dynamics model
                actually a concatenation of (1) current obs, and (K-1) past obs
            step_number:
                which step number the rollout is currently on
                (used to calculate costs)
            actions_taken_so_far:
                used to restore state of the env to correct place,
                when using ground-truth dynamics
            starting_fullenvstate:
                full state of env before this rollout, used for env resets
                (when using ground-truth dynamics)

        Returns:
            best_action: optimal action to perform, according to controller
            resulting_states_ls: predicted results of executing the candidate
                                 action sequences
        """

        ####################################################################
        # sample N random candidate action sequences, each of length horizon
        ####################################################################
        np.random.seed()  # get different action samples for each rollout

        all_samples = []
        for _ in range(self.N):
            sample_per_traj = []
            for _ in range(self.horizon):
                sample_per_traj.append(self.rand_policy.get_action(
                    curr_state,
                    random_sampling_params=self.random_sampling_params,
                    hold_action_overrideToOne=True))
            all_samples.append(np.array(sample_per_traj))
        # all_actions: [num_sample_seq, num_rollout, num_action_dim]
        all_actions = np.array(all_samples)

        #############################################################################
        # have model predict the result of executing those candidate action sequences
        #############################################################################

        # [horizon+1, N, state_size]
        resulting_states_ls = self.dyn_model.do_forward_sim(curr_state, np.copy(all_actions))

        #####################################
        # evaluate the predicted trajectories
        # calculate costs
        #####################################

        # average all the ending states in the recording as goal position
        goal_pos = get_goal_pos()
        costs = calculate_costs(resulting_states_ls, goal_pos, self.cost_fn)

        # pick best action sequence
        best_score = np.min(costs)
        best_sim_number = np.argmin(costs)
        best_sequence = all_actions[best_sim_number]
        best_action = np.copy(best_sequence[0])

        # execute the candidate action sequences on the real dynamics
        # instead just on the model

        return best_action


def calculate_costs(resulting_states_ls, goal, cost_fn):
    """
    Rank various predicted trajectories (by cost)

    Args:
        resulting_states_ls:
            predicted trajectories [horizon, N, state_size]
        actions:
            the actions that were "executed" in order to achieve the predicted trajectories
            [N, h, action_size]
        reward_func:
            calculates the rewards associated with each state transition in the predicted trajectories

    Returns:
        cost_for_ranking: cost associated with each candidate action sequence [N,]
    """

    ###########################################################
    # calculate costs associated with each predicted trajectory
    ###########################################################

    #init vars for calculating costs
    horizon, num_sample_seq, _ = resulting_states_ls.shape
    costs = np.zeros((num_sample_seq * len(resulting_states_ls),))

    # accumulate cost over each timestep
    costs = []
    for traj in range(num_sample_seq):
        cost = 0
        for h in range(horizon-1, 0, -1):
            cost = cost_fn(resulting_states_ls[h, traj, :], goal) + 0.9 * cost
        costs.append(cost)

    return np.array(costs)
