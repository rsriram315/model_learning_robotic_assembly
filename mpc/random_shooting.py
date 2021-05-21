# flake8: noqa
import numpy as np
from copy import deepcopy


class RandomShooting(object):
    def __init__(self, env, dyn_model, reward_fn, rand_policy, params):
        self.K = params.K
        self.horizon = params.horizon
        self.N = params.num_control_samples  # number of random action sequences
        self.dyn_model = dyn_model
        self.reward_fn = reward_fn
        self.rand_policy = rand_policy
        self.env = deepcopy(env)

        # TODO position range or even also rotation range
        self.random_sampling_params = {'sample_pos': params.rand_policy_sample_pos,
                                       'pos_min': params.rand_policy_pos_min,
                                        'pos_max': params.rand_policy_vel_max,
                                        'hold_action': params.rand_policy_hold_action}

    def get_action(self, curr_state_K, actions_taken_so_far):
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

        # sample N random candidate action sequences, each of length horizon
        np.random.seed()  # get different action samples for each rollout

        all_samples = []
        junk = 1
        for _ in range(self.N):
            sample_per_traj = []
            for _ in range(self.horizon):
                sample_per_traj.append(self.rand_policy.get_action(junk, pre_action=None,
                                                                   random_sampling_params=self.random_sampling_params,
                                                                   hold_action_overrideToOne=True)[0])
            all_samples.append(np.array(sample_per_traj))
        all_samples = np.array(all_samples)

        # make each action element be (past K actions) instead of just (curr action)

        # all samples: [N, horizon, action_dim]
        all_actions = turn_actions_into_actions_K(actions_taken_so_far, all_samples, self.K, self.N, self.horizon)
        # all_actions: [N, horizon, K, action_dim]

        # have model predict the result of executing those candidate action sequences
        # [horizon+1, N, state_size]
        resulting_states_ls = self.dyn_model.do_forward_sim([curr_state_K, 0], np.copy(all_actions))

        # evaluate the predicted trajectories
        # calculate costs
        costs, mean_costs, std_costs = calculate_costs(resulting_states_ls, all_samples, self.reward_fn)

        # pick best action sequence
        best_score = np.min(costs)
        best_sim_number = np.argmin(costs)
        best_sequence = all_actions[best_sim_number]
        best_action = np.copy(best_sequence[0])

        # execute the candidate action sequences on the real dynamics
        # instead just on the model

        return best_action, resulting_states_ls


def turn_actions_into_actions_K(actions_taken_so_far, all_samples, K, N, horizon):
    """
    start with array, where each entry is (a_t)
    end with array, where each entry is (a_{t-(K-1) ..., a_{t-1}, a_{t}})
    """
    pass


def calculate_costs(resulting_states_ls, actions, reward_func,
                    evaluating, take_exploratory_actions):
    """
    Rank various predicted trajectories (by cost)

    Args:
        resulting_states_ls:
            predicted trajectories [horizon + 1, N, state_size]
        actions:
            the actions that were "executed" in order to achieve the predicted trajectories
            [N, h, action_size]
        reward_func:
            calculates the rewards associated with each state transition in the predicted trajectories

    Returns:
        cost_for_ranking: cost associated with each candidate action sequence [N,]
    """

    pass