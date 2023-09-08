# flake8: noqa
import numpy as np
from scipy.spatial.transform import Rotation as R
from mpc.helper import calculate_costs
import copy
_FLOAT_EPS = np.finfo(np.float64).eps


class MPPI:
    """
    Generate multiple random actions using Model Predictive Path Integral
    Select the best action
    """
    def __init__(self, env, dyn_model, cost_fn, rand_policy, params):
        self.env = env
        self.dyn_model = dyn_model
        self.cost_fn = cost_fn
        self.rand_policy = rand_policy
        print(params)
        self.horizon = params.horizon
        self.N = params.num_sample_seq

        # init mppi vars
        self.mppi_gamma = params.mppi_gamma
        self.action_dim = 15  # pos, force, rot_mat
        self.sigma = params.mppi_mag_noise * np.ones(self.action_dim)
        self.beta = params.mppi_beta
        self.mppi_mean = None
        self.angle_min = params.rand_policy_angle_min
        self.angle_max = params.rand_policy_angle_max

        self.counter = 1
    ##########################################################
    # update action mean using weighted average of the actions
    ##########################################################
    def mppi_update(self, scores, all_samples):

        #######################################################
        # how each simulation's score compare to the best score
        #######################################################
        weights = np.exp(-self.mppi_gamma * scores)[:, None, None]  # [N, 1, 1] # mppi formulation
        sum_weights = np.sum(weights) + _FLOAT_EPS  # numerical stability

        #######################################################################
        # weight all actions of the sequence by that sequence's resulted reward
        #######################################################################
        weighted_actions = np.copy(all_samples)
        weighted_actions[:, :, :3] = weights * all_samples[:, :, :3]  # [N, H, action_dim]
        self.mppi_mean[:, :3] = np.sum(weighted_actions[:, :, :3], axis=0) / sum_weights
        
        # sum over all the sampled trajectories
        # weighted_actions = weights * all_samples  # [N, H, action_dim]
        for h in range(all_samples.shape[1]):
            certain_horizon = np.copy(all_samples[:, h, 6:15]).reshape((-1, 3, 3))
            rot_certain_horizon = R.from_matrix(certain_horizon)
            w = np.squeeze(weights/sum_weights, axis=(1,2))
            self.mppi_mean[h, 6:15] = rot_certain_horizon.mean(weights=w).as_matrix().flatten()
        # the generated mppi_mean is a weighted average trajectories of the sampled trajectories
        # return 1st element of the mean, which corresponds to curr timestep
        return self.mppi_mean[0]

    def get_action(self, curr_state, goal_state, step):
        """
        Select optimal action

        Agrs:
            curr_state:
                current "state" as known by the dynamics model
        Returns:
            best_action: optimal action to perform, according to controller
        """
        if self.counter == 1:
            # get the starting pose rotations in terms of euler angles
            self.init_rot = R.from_matrix(curr_state[6:15].reshape(3,3))
            self.init_euler_angle = self.init_rot.as_euler('zyx')
            print("self.init_euler_angle", (self.init_euler_angle) * 180 / np.pi)
        # past action is the first step of the prev averaged trajectory
        past_action = np.copy(self.mppi_mean[0])  # mu_{t-1}

        # remove the 1st entry of mean (mean from last timestamp, which was just executed)
        # and copy the penultimate entry to the last entry (starting point, for the next timestep)
        self.mppi_mean[:-1] = self.mppi_mean[1:]  # mu_{t}

        for k in range(1):
            ##############################################
            # noise source
            ##############################################
            # sample position
            rand_set_point = np.random.uniform(np.array([-1, -1, -1]), np.array([1, 1, 1]),
                                size=(self.N, self.horizon, 3)) * self.sigma[:3]
            rand_force = np.zeros((self.N, self.horizon, 3))
            # noisy rotation matrix
            rand_euler_delta = np.random.uniform(self.angle_min, self.angle_max, size=(self.N, self.horizon, 3)) * np.pi
            euler_init = np.tile(self.init_euler_angle, (self.N, self.horizon, 1))
            # add rotation noise to the initial Euler angle and covert to rotation matrix
            rand_euler_raw  = euler_init + rand_euler_delta
            rand_rot = np.stack([R.from_euler('zyx', euler).as_matrix().reshape((-1, 9))
                                    for euler in rand_euler_raw], axis=0)
            # concatenate all sampled noise
            eps = np.concatenate((rand_set_point, rand_force, rand_rot), axis=2)
            all_samples = copy.deepcopy(eps)

            # actions = mean + noise, then smooth the actions temporally
            for h in range(self.horizon):
                if h == 0:
                    # first step, the past action and mppi_mean are just zero ,so the
                    # first generate action (1st horizon) is just the noise itself
                    all_samples[:, h, :3] = (self.beta * eps[:, h, :3]) + past_action[:3]
                    new_rot = [eps_rot.reshape((3, 3)) for eps_rot in eps[:, h, 6:15]]
                    past_rot = past_action[6:15].reshape((3, 3))

                    for n in range(self.N):
                        interp_R = R.from_matrix([new_rot[n], past_rot])
                        all_samples[n, h, 6:15] = interp_R.mean([self.beta, 1]).as_matrix().flatten()

                else:
                    all_samples[:, h, :3] = self.beta*(self.mppi_mean[h, :3] + eps[:, h, :3]) + (1-self.beta)*all_samples[:, h-1, :3]
                    new_rot = [eps_rot.reshape((3, 3)) for eps_rot in eps[:, h, 6:15]]
                    past_rot = all_samples[:, h-1, 6:15].reshape((-1, 3, 3))

                    for n in range(self.N):
                        interp_R = R.from_matrix([new_rot[n], past_rot[n]])
                        all_samples[n, h, 6:15] = interp_R.mean([self.beta, 1-self.beta]).as_matrix().flatten()
            
            # resulting candidate action sequences, all_samples: [N, horizon, action_dim]
            all_samples = np.clip(all_samples, -1, 1)

            #####################################################################
            # model predict results of executing those candidate action sequences
            #####################################################################

            # [horizon+1, N, state_size]
            resulting_states_ls, norm_resulting_states_ls= \
                self.dyn_model.do_forward_sim(curr_state, np.copy(all_samples))

            # average all the ending states in the recording as goal position
            costs = calculate_costs(resulting_states_ls, goal_state, self.cost_fn)
            # best_seq_num = np.argmin(costs)
            # use all paths to update action mean (for horizon steps)
            # selected_action = self.mppi_update(np.atleast_1d(costs[best_seq_num]), all_samples[best_seq_num][None, :])
            selected_action = self.mppi_update(costs, all_samples)

        self.counter += 1
        selected_action = self.dyn_model.norm.inv_normalize(selected_action[None, None, :], is_action=True)[0]
        pred_state = np.squeeze(self.dyn_model.do_forward_sim(curr_state, np.copy(selected_action[None, None, : ]))[0])
        return selected_action, pred_state
