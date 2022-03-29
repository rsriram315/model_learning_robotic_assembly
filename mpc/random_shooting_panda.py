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
        # all_actions: [num_sample_seq, num_rollout, num_action_dim]
        all_actions = np.array(all_samples)
        # print("all samples: ", all_actions[:5,:,:])

        # z_set_point_ls = np.arange(curr_state[2]+0.005, curr_state[2]-0.005, -0.0001)
        # # print("z set points", z_set_point_ls)
        # for z in z_set_point_ls:
        #     action = np.copy(curr_state)
        #     action[2] = z 
        #     action[6:15] = (R.from_quat([0.973, -0.226, -0.041, 0.007])).as_matrix().flatten()
        #     norm_action = self.dyn_model.norm.normalize(action[None, None, :])
        #     all_samples.append(norm_action)
        # all_actions = np.array(all_samples)
        # # print("all actions", all_actions[:5])
        if self.step == 500:
            all_samples = []
            z_set_point_ls = np.arange(curr_state[2]+0.01, curr_state[2]-0.01, -0.0001)
            # print("z set points", z_set_point_ls)
            for z in z_set_point_ls:
                action = np.copy(curr_state)
                action[2] = z 
                # action[6:15] = (R.from_quat([0.973, -0.226, -0.041, 0.007])).as_matrix().flatten()
                norm_action = self.dyn_model.norm.normalize(action[None, None, :])
                all_samples.append(norm_action)
            all_actions = np.array(all_samples)

            # plot to check sampled actions
            x_axis = [index for index in range (all_actions.shape[0])]
            y1_axis = all_actions[:,0,0]
            y2_axis = all_actions[:,0,1]
            y3_axis = all_actions[:,0,2]
            norm_curr_state = np.squeeze(self.dyn_model.norm.normalize(curr_state[None, None, :])[0])
            import matplotlib.pyplot as plt
            plt.subplot(1,3,1)
            plt.scatter(x_axis, y1_axis, marker="o", color="green")
            plt.plot(x_axis, [norm_curr_state[0] for _ in range (all_actions.shape[0])], marker="o", color="yellow")
            plt.xlabel('samples')
            plt.ylabel('sampled x points')
            plt.subplot(1,3,2)
            plt.scatter(x_axis, y2_axis, marker="o", color="red")
            plt.plot(x_axis, [norm_curr_state[1] for _ in range (all_actions.shape[0])], marker="o", color="yellow")
            plt.xlabel('samples')
            plt.ylabel('sampled y points')
            plt.subplot(1,3,3)
            plt.scatter(x_axis, y3_axis, marker="o", color="blue")
            plt.plot(x_axis, [norm_curr_state[2] for _ in range (all_actions.shape[0])], marker="o", color="yellow")
            plt.xlabel('samples')
            plt.ylabel('sampled z points')
            plt.figure()

        #############################################################################
        # have model predict the result of executing those candidate action sequences
        #############################################################################

        # [horizon+1, N, state_size]
        resulting_states_ls, norm_resulting_states_ls = self.dyn_model.do_forward_sim(curr_state, np.copy(all_actions))
        # print("resulting_states_ls", resulting_states_ls[0, :5, :])
        if self.step == 500:
            # ploting to see resulting statesplot
            import matplotlib.pyplot as plt
            x_axis = [index for index in range (norm_resulting_states_ls.shape[1])]
            norm_curr_state = np.squeeze(self.dyn_model.norm.normalize(curr_state[None, None, :])[0])
            y1_axis = norm_resulting_states_ls[0,:,0]
            y2_axis = norm_resulting_states_ls[0,:,1]
            y3_axis = norm_resulting_states_ls[0,:,2]        
            plt.subplot(1,3,1)
            plt.scatter(x_axis, y1_axis, marker="o", color="green")
            plt.plot(x_axis, [norm_curr_state[0] for _ in range (all_actions.shape[0])], marker="o", color="yellow")
            plt.xlabel('samples')
            plt.ylabel('resulting state x normalized')
            plt.subplot(1,3,2)
            plt.scatter(x_axis, y2_axis, marker="o", color="red")
            plt.plot(x_axis, [norm_curr_state[1] for _ in range (all_actions.shape[0])], marker="o", color="yellow")
            plt.xlabel('samples')
            plt.ylabel('resulting state y normalized')
            plt.subplot(1,3,3)
            plt.scatter(x_axis, y3_axis, marker="o", color="blue")
            plt.plot(x_axis, [norm_curr_state[2] for _ in range (all_actions.shape[0])], marker="o", color="yellow")
            plt.xlabel('samples')
            plt.ylabel('resulting state z normalized')
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

        #####################################
        # evaluate the predicted trajectories
        # calculate costs
        #####################################
        costs = calculate_costs(resulting_states_ls, goal_state, self.cost_fn)
        
        # pick best action sequence
        best_sim_number = np.argmin(costs)
        best_sequence = all_actions[best_sim_number]
        best_action = np.copy(best_sequence[0])
        # print(best_action.shape, best_action[None, None, :].shape, best_sequence.shape)
        # print("best action before inv_normalizing", best_action)
        
        # execute the candidate action sequences on the real dynamics instead just on the model
        # unnormalized best actions
        best_action = self.dyn_model.norm.inv_normalize(best_action[None, None, :], is_action=True)[0]
        # print("best action after inv_normalizing", best_action)
        pred_state = resulting_states_ls[0,best_sim_number,:]
        # norm_pred_states = norm_resulting_states_ls[0,best_sim_number,:]

        self.step += 1

        return best_action, pred_state #, norm_pred_states
