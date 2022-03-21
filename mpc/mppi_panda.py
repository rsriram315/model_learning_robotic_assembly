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

        self.horizon = params.horizon
        self.N = params.num_sample_seq

        # init mppi vars
        self.mppi_gamma = params.mppi_gamma
        self.action_dim = 15  # pos, force, rot_mat
        self.sigma = params.mppi_mag_noise * np.ones(self.action_dim)
        self.beta = params.mppi_beta
        self.mppi_mean = None
        # self.mppi_mean_normalized  = None
        # self.mppi_mean = np.zeros((self.horizon, self.action_dim))

        self.counter = 1
    ##########################################################
    # update action mean using weighted average of the actions
    # (by their resulting scores), Eq.2 on my slides
    ##########################################################
    def mppi_update(self, scores, all_samples):

        #######################################################
        # how each simulation's score compare to the best score
        #######################################################
        # TODO: why subtract the max of scores?
        # because scores here are -costs, therefore,
        # scores - max(scores) would yied scores (-inf, 0]
        # weights = np.exp(self.mppi_gamma * (scores - np.amax(scores)))[:, None, None]  # [N, 1, 1] # pddm formulation
        weights = np.exp(-self.mppi_gamma * scores)[:, None, None]  # [N, 1, 1] # mppi formulation
        sum_weights = np.sum(weights) + _FLOAT_EPS  # numerical stability
        print("weigths", weights)
        #######################################################################
        # weight all actions of the sequence by that sequence's resulted reward
        #######################################################################
        weighted_actions = np.copy(all_samples)
        weighted_actions[:, :, :3] = weights * all_samples[:, :, :3]  # [N, H, action_dim]
        self.mppi_mean[:, :3] = np.sum(weighted_actions[:, :, :3], axis=0) / sum_weights
        
        # sum over all the sampled trajectories
        # weighted_actions = weights * all_samples  # [N, H, action_dim]
        # self.mppi_mean = np.sum(weighted_actions, axis=0) / sum_weights

        for h in range(all_samples.shape[1]):
            certain_horizon = np.copy(all_samples[:, h, 6:15]).reshape((-1, 3, 3))
            rot_certain_horizon = R.from_matrix(certain_horizon)
            w = np.squeeze(weights/sum_weights, axis=(1,2))
            self.mppi_mean[h, 6:15] = rot_certain_horizon.mean(weights=w).as_matrix().flatten()
        # the generated mppi_mean is a weighted average trajectories of the sampled trajectories
        # return 1st element of the mean, which corresponds to curr timestep
        print(" \n self.mppi_mean[:, :3]",  self.mppi_mean[:, :3])
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
            self.init_rot = R.from_matrix(curr_state[6:15].reshape(3,3))
            self.init_euler_angle = self.init_rot.as_euler('zyx')
            print("self.init_euler_angle", (self.init_euler_angle) * 180 / np.pi)
        # past action is the first step of the prev averaged trajectory
        past_action = np.copy(self.mppi_mean[0])  # mu_{t-1}
        print("past_action", past_action[:3])
        # past_action_normalized = np.squeeze(self.dyn_model.norm.normalize(past_action[None, None, :], is_action=True))

        # remove the 1st entry of mean (mean from last timestamp, which was just executed)
        # and copy the penultimate entry to the last entry (starting point, for the next timestep)
        self.mppi_mean[:-1] = self.mppi_mean[1:]  # mu_{t}
        # self.mppi_mean_normalized = self.dyn_model.norm.normalize(self.mppi_mean[None,:,:], is_action=True, axis=0)
        print("mppi_mean", self.mppi_mean[:, :3])
        # print("mppi_mean_normalized", self.mppi_mean_normalized)
        for k in range(1):
            ##############################################
            # noise source
            ##############################################
            # only disturb set point position for prototyping
            rand_set_point = np.random.uniform(np.array([-1, -1, -1]), np.array([1, 1, 1]),
                                size=(self.N, self.horizon, 3)) * self.sigma[:3]
            rand_force = np.zeros((self.N, self.horizon, 3))
            # noisy rotation matrix
            rand_euler_delta = np.random.uniform(-0.02, 0.02, size=(self.N, self.horizon, 3)) * np.pi
            # rand_euler_delta = np.random.normal(loc=0, scale=0.02/3, size=(self.N, self.horizon, 3)) * np.pi
            euler_init = np.tile(self.init_euler_angle, (self.N, self.horizon, 1))
            rand_euler_raw  = euler_init + rand_euler_delta
            rand_rot = np.stack([R.from_euler('zyx', euler).as_matrix().reshape((-1, 9))
                                    for euler in rand_euler_raw], axis=0)
            print("init euler", self.init_rot.as_euler('zyx', degrees=True))
            print("rand euler delta", rand_euler_delta[0,0]*180/ np.pi)
            print("rand euler", rand_euler_raw[0,0]*180/ np.pi)
            eps = np.concatenate((rand_set_point, rand_force, rand_rot), axis=2)
            all_samples = copy.deepcopy(eps)
            print("all samples before smoothing", all_samples[:10,0,:3])

            # actions = mean + noise, then smooth the actions temporally
            # TODO: where is the beta * (action_mean + noise) from?
            # in the slides, it should be just beta * noise! Double check is needed!!!


            # for h in range(self.horizon):
            #     if h == 0:
            #         # first step, the past action and mppi_mean are just zero ,so the
            #         # first generate action (1st horizon) is just the noise itself
            #         # all_samples[:, h, :3] = \
            #         #     (self.beta * (self.mppi_mean[h, :3] + eps[:, h, :3]) +
            #         #     (1 - self.beta) * past_action[:3])
            #         all_samples[:, h, :3] = (self.beta * eps[:, h, :3]) + past_action[:3]
                    
                    
            #         # new_rot = [eps_rot.reshape((3, 3)) @ self.mppi_mean[h, 6:15].reshape(3,3) for eps_rot in eps[:, h, 6:15]]
            #         new_rot = [eps_rot.reshape((3, 3)) for eps_rot in eps[:, h, 6:15]]
            #         past_rot = past_action[6:15].reshape((3, 3))

            #         for n in range(self.N):
            #             interp_R = R.from_matrix([new_rot[n], past_rot])
            #             # all_samples[n, h, 6:15] = interp_R.mean([self.beta, 1-self.beta]).as_matrix().flatten()
            #             all_samples[n, h, 6:15] = interp_R.mean([self.beta, 1]).as_matrix().flatten()

            #     else:
            #         # all_samples[:, h, :3] = \
            #         #     (self.beta * (self.mppi_mean[h, :3] + eps[:, h, :3]) +
            #         #     (1 - self.beta) * all_samples[:, h-1, :3])
            #         all_samples[:, h, :3] = (self.beta * eps[:, h, :3]) + (1 - self.beta) * all_samples[:, h-1, :3] + self.mppi_mean[h, :3]
                    
                    
            #         # new_rot = [eps_rot.reshape((3, 3)) @ self.mppi_mean[h, 6:15].reshape(3,3) for eps_rot in eps[:, h, 6:15]]
            #         new_rot = [eps_rot.reshape((3, 3)) for eps_rot in eps[:, h, 6:15]]
            #         past_rot = all_samples[:, h-1, 6:15].reshape((-1, 3, 3))

            #         for n in range(self.N):
            #             interp_R = R.from_matrix([new_rot[n], past_rot[n]])
            #             all_samples[n, h, 6:15] = interp_R.mean([self.beta, 1-self.beta]).as_matrix().flatten()


            for h in range(self.horizon):
                if h == 0:
                    # first step, the past action and mppi_mean are just zero ,so the
                    # first generate action (1st horizon) is just the noise itself
                    # all_samples[:, h, :3] = \
                    #     (self.beta * (self.mppi_mean[h, :3] + eps[:, h, :3]) +
                    #     (1 - self.beta) * past_action[:3])
                    all_samples[:, h, :3] = (self.beta * eps[:, h, :3]) + past_action[:3]
                    
                    
                    # new_rot = [eps_rot.reshape((3, 3)) @ self.mppi_mean[h, 6:15].reshape(3,3) for eps_rot in eps[:, h, 6:15]]
                    new_rot = [eps_rot.reshape((3, 3)) for eps_rot in eps[:, h, 6:15]]
                    past_rot = past_action[6:15].reshape((3, 3))

                    for n in range(self.N):
                        interp_R = R.from_matrix([new_rot[n], past_rot])
                        # all_samples[n, h, 6:15] = interp_R.mean([self.beta, 1-self.beta]).as_matrix().flatten()
                        all_samples[n, h, 6:15] = interp_R.mean([self.beta, 1]).as_matrix().flatten()

                else:
                    # all_samples[:, h, :3] = \
                    #     (self.beta * (self.mppi_mean[h, :3] + eps[:, h, :3]) +
                    #     (1 - self.beta) * all_samples[:, h-1, :3])
                    # all_samples[:, h, :3] = self.beta * eps[:, h, :3] + (1 - self.beta) * (all_samples[:, h-1, :3] - self.mppi_mean[h-1, :3]) + self.mppi_mean[h, :3]
                    all_samples[:, h, :3] = self.beta*(self.mppi_mean[h, :3] + eps[:, h, :3]) + (1-self.beta)*all_samples[:, h-1, :3]
                    
                    # new_rot = [eps_rot.reshape((3, 3)) @ self.mppi_mean[h, 6:15].reshape(3,3) for eps_rot in eps[:, h, 6:15]]
                    new_rot = [eps_rot.reshape((3, 3)) for eps_rot in eps[:, h, 6:15]]
                    past_rot = all_samples[:, h-1, 6:15].reshape((-1, 3, 3))

                    for n in range(self.N):
                        interp_R = R.from_matrix([new_rot[n], past_rot[n]])
                        all_samples[n, h, 6:15] = interp_R.mean([self.beta, 1-self.beta]).as_matrix().flatten()
            # resulting candidate action sequences, all_samples: [N, horizon, action_dim]
            print("all samples after smoothing: ", all_samples[:10,0,:3])
            all_samples = np.clip(all_samples, -1, 1)
            print("all samples after clipping: ", all_samples[:10,0,:3])
            
            #####################
            # plot for debugging
            #####################
            if self.counter == 1000:
                all_samples = []
                z_set_point_ls = np.arange(curr_state[2]+0.005, curr_state[2]-0.005, -0.0001)
                # print("z set points", z_set_point_ls)
                for z in z_set_point_ls:
                    action = np.copy(curr_state)
                    action[2] = z 
                    action[6:15] = (R.from_quat([0.973, -0.226, -0.041, 0.007])).as_matrix().flatten()
                    norm_action = self.dyn_model.norm.normalize(action[None, None, :])
                    all_samples.append(norm_action)
                all_samples = np.array(all_samples)
                # print("all actions", all_actions[:5])
                # plot to check sampled actions
                x_axis = [index for index in range (all_samples.shape[0])]
                y1_axis = all_samples[:,0,0]
                y2_axis = all_samples[:,0,1]
                y3_axis = all_samples[:,0,2]
                norm_curr_state = np.squeeze(self.dyn_model.norm.normalize(curr_state[None, None, :])[0])
                import matplotlib.pyplot as plt
                plt.subplot(1,3,1)
                plt.scatter(x_axis, y1_axis, marker="o", color="green")
                plt.plot(x_axis, [norm_curr_state[0] for _ in range (all_samples.shape[0])], marker="o", color="yellow")
                plt.xlabel('samples')
                plt.ylabel('sampled x points')
                plt.subplot(1,3,2)
                plt.scatter(x_axis, y2_axis, marker="o", color="red")
                plt.plot(x_axis, [norm_curr_state[1] for _ in range (all_samples.shape[0])], marker="o", color="yellow")
                plt.xlabel('samples')
                plt.ylabel('sampled y points')
                plt.subplot(1,3,3)
                plt.scatter(x_axis, y3_axis, marker="o", color="blue")
                plt.plot(x_axis, [norm_curr_state[2] for _ in range (all_samples.shape[0])], marker="o", color="yellow")
                plt.xlabel('samples')
                plt.ylabel('sampled z points')
                plt.figure()


            #####################################################################
            # model predict results of executing those candidate action sequences
            #####################################################################

            # [horizon+1, N, state_size]
            resulting_states_ls, norm_resulting_states_ls= \
                self.dyn_model.do_forward_sim(curr_state, np.copy(all_samples))
            print("resulting_states_ls", resulting_states_ls[0, :5, :3])
            
            #####################
            # plot for debugging
            #####################
            if self.counter == 1000:
                # ploting to see resulting statesplot
                import matplotlib.pyplot as plt
                x_axis = [index for index in range (norm_resulting_states_ls.shape[1])]
                norm_curr_state = np.squeeze(self.dyn_model.norm.normalize(curr_state[None, None, :])[0])
                y1_axis = norm_resulting_states_ls[0,:,0]
                y2_axis = norm_resulting_states_ls[0,:,1]
                y3_axis = norm_resulting_states_ls[0,:,2]
                plt.subplot(1,3,1)
                plt.scatter(x_axis, y1_axis, marker="o", color="green")
                plt.plot(x_axis, [norm_curr_state[0] for _ in range (all_samples.shape[0])], marker="o", color="yellow")
                plt.xlabel('samples')
                plt.ylabel('resulting state x ')
                plt.subplot(1,3,2)
                plt.scatter(x_axis, y2_axis, marker="o", color="red")
                plt.plot(x_axis, [norm_curr_state[1] for _ in range (all_samples.shape[0])], marker="o", color="yellow")
                plt.xlabel('samples')
                plt.ylabel('resulting state y')
                plt.subplot(1,3,3)
                plt.scatter(x_axis, y3_axis, marker="o", color="blue")
                plt.plot(x_axis, [norm_curr_state[2] for _ in range (all_samples.shape[0])], marker="o", color="yellow")
                plt.xlabel('samples')
                plt.ylabel('resulting state z')
                plt.figure()

                # ploting to see resulting statesplot
                import matplotlib.pyplot as plt
                x_axis = [index for index in range (resulting_states_ls.shape[1])]
                curr_state = [curr_state for _ in range (resulting_states_ls.shape[1])]
                curr_state = np.asarray(curr_state)
                y1_axis = resulting_states_ls[0,:,0]
                y2_axis = resulting_states_ls[0,:,1]
                y3_axis = resulting_states_ls[0,:,2]
                plt.subplot(1,3,1)
                plt.scatter(x_axis, y1_axis, marker="o", color="green")
                plt.plot(x_axis, curr_state[:,0], marker="o", color="yellow")
                plt.xlabel('samples')
                plt.ylabel('resulting state x ')
                plt.subplot(1,3,2)
                plt.scatter(x_axis, y2_axis, marker="o", color="red")
                plt.plot(x_axis, curr_state[:,1], marker="o", color="yellow")
                plt.xlabel('samples')
                plt.ylabel('resulting state y')
                plt.subplot(1,3,3)
                plt.scatter(x_axis, y3_axis, marker="o", color="blue")
                plt.plot(x_axis, curr_state[:,2], marker="o", color="yellow")
                plt.xlabel('samples')
                plt.ylabel('resulting state z')
                plt.show()


            # average all the ending states in the recording as goal position
            costs = calculate_costs(resulting_states_ls, goal_state, self.cost_fn)
            idx  = np.argsort(costs)
            sorted_cost = costs[idx]
            sorted_all_samples = all_samples[idx]
            best_seq_num = np.argmin(costs)
            print("costs", costs[:5])
            # use all paths to update action mean (for horizon steps)
            print("\n", sorted_cost[:20].shape, sorted_all_samples.shape)
            selected_action = self.mppi_update(np.atleast_1d(costs[best_seq_num]), all_samples[best_seq_num][None, :])
            # selected_action = self.mppi_update(np.atleast_1d(sorted_cost[0]), sorted_all_samples[0][None, :])
            print("selected_action",selected_action)

        self.counter += 1
        # print("selected action before inv normalizing", selected_action)
        selected_action = self.dyn_model.norm.inv_normalize(selected_action[None, None, :], is_action=True)[0]
        print("selected action after inv normalizing", selected_action)
        # selected_action[6:15] = np.eye(3).flatten()
        pred_state = np.squeeze(self.dyn_model.do_forward_sim(curr_state, np.copy(selected_action[None, None, : ]))[0])
        return selected_action, pred_state
