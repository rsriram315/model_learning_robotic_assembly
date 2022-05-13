# flake8: noqa
from time import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from mpc.helper import calculate_costs
import time
class RandomShooting:
    """
    Generate multiple random action rollouts, and select the best one
    """
    def __init__(self, env, dyn_model, cost_fn, rand_policy, params):
        self.env = env
        self.dyn_model = dyn_model
        self.cost_fn = cost_fn
        self.rand_policy = rand_policy
        self.horizon = params.horizon
        self.N = params.num_sample_seq  # number of random action sequences
        self.random_sampling_params = {'angle_min': params.rand_policy_angle_min,
                                       'angle_max': params.rand_policy_angle_max,
                                       'hold_action': params.rand_policy_hold_action}
        self.step = 0
    def get_action(self, curr_state, goal_state, step):
        """
        Select optimal action

        Agrs:
            curr_state:
                current "state" as known by the dynamics model
        Returns:
            best_action: optimal action to perform, according to controller
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
                    hold_action_overrideToOne=True,
                    traj_count=step,))
            all_samples.append(np.array(sample_per_traj))
        all_actions = np.array(all_samples)

        #############################################################################
        # have model predict the result of executing those candidate action sequences
        #############################################################################

        # [horizon+1, N, state_size]
        resulting_states_ls, norm_resulting_states_ls = self.dyn_model.do_forward_sim(curr_state, np.copy(all_actions))
        
        #####################################
        # evaluate the predicted trajectories
        # calculate costs
        #####################################
        costs = calculate_costs(resulting_states_ls, goal_state, self.cost_fn)
        
        # pick best action sequence
        best_sim_number = np.argmin(costs)
        best_sequence = all_actions[best_sim_number]
        best_action = np.copy(best_sequence[0])

        
        # execute the candidate action sequences on the real dynamics instead just on the model
        # unnormalized best actions
        best_action = self.dyn_model.norm.inv_normalize(best_action[None, None, :], is_action=True)[0]
        pred_state = resulting_states_ls[0,best_sim_number,:]

        self.step += 1

        return best_action, pred_state #, norm_pred_states
