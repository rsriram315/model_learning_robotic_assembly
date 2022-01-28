# flake8: noqa
import time
import numpy as np
# import robosuite.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R
from mpc.random_shooting_panda import RandomShooting
from mpc.mppi_panda import MPPI
import matplotlib.pyplot as plt
from utils.data_collection_node import DataCollection
class MPCRollout:
    def __init__(self, env, dyn_model, rand_policy, goal_state, params):
        self.env = env
        self.dyn_model = dyn_model
        self.max_step = params.max_step
        self.controller_type = params.controller_type
        self.goal_state = goal_state

        # init controller parameters
        if params.controller_type == 'rand_shooting':
            self.controller = RandomShooting(self.env,
                                             self.dyn_model,
                                             self.cost,
                                             rand_policy, params)
        elif params.controller_type == 'mppi':
            params.mppi_gamma = 300
            params.mppi_mag_noise = 0.9
            params.mppi_beta = 0.9
            self.controller = MPPI(self.env,
                                   self.dyn_model,
                                   self.cost,
                                   rand_policy, params)
    
    def cost(self, curr_state, goal_state ):
        """
        cost functin to evaluate sampled trajectories
        """
        # reward is normalized
        mpc_cost = self.env._cost(pred_state=curr_state, goal_state=goal_state)
        return mpc_cost
    
    def perform_rollout(self, starting_envstate):
        """
        Args:
            starting_envstate: state of the mujoco env (enough to allow resetting it)
            starting_observation: obs returned by env.reset when the state itself was starting_fullenvstate
            controller_type: 'rand' or 'mppi'

        Populates:
            traj_take: list of (T+1) states visited
            actions_take: list of (T) actions taken
            total_reward_for_episode: sum of rewards for this rollout

        Returns:
            rollout_info: saving all info relevant to this rollout
        """
        rollout_start = time.time()

        #######################
        # init vars for rollout
        #######################
        step = 1
        # self.starting_envstate = starting_envstate

        ###################################
        # initialize first K states/actions
        ###################################
        curr_state = np.copy(starting_envstate)

        #######################################
        # loop over steps in rollout
        #######################################
        done = False
        count = 0

        true_state = []
        predicted_state = []
        norm_predicted_state = []
        max_input = []


        self.data_dir = "/home/rsr7rng/thesis/model_learning_panda/recorded_data"
        self.data_collector = DataCollection(self.env, self.data_dir)
        self.data_collector.reset()

        while not(done or step >= self.max_step):
            count += 1
            # get optimal action
            if self.controller_type == 'mppi' and count == 1:
                self.controller.mppi_mean = np.tile(curr_state, (self.controller.horizon, 1))
                # self.controller.mppi_mean[:, :3] = [0, 0, 0]
                # self.controller.mppi_mean[:, 3:6] = [0, 0, 0]  # force action should be zero
                # self.controller.mppi_mean[:, 6:15] = np.eye(3,3).flatten()  # action rotation is delta
            print("current state before executing mpc", self.env._get_obs()[:3])
            
            best_action, pred_next_state = self.controller.get_action(curr_state, self.goal_state, step)

            predicted_state.append(pred_next_state)
            # norm_predicted_state.append(norm_pred_next_state)
            
            print("best_action", best_action[:3])
            best_action_pos = best_action[:3] - np.copy(self.env._get_obs()[:3])
            max_input.append(best_action_pos)
            best_action_rot = best_action[6:]
            
            action_to_take = np.hstack((best_action_pos,
                                       best_action_rot))
            print("current state before taking action:", self.env._get_obs()[:3])
            
            ########################################################################    
            # recording current tcp state and action sent to cartesisan impedence setpoint
            ########################################################################
            curr_pose = np.array([
                self.env.robot_interface._arm_state.tcp_pose_base.position.x,
                self.env.robot_interface._arm_state.tcp_pose_base.position.y,
                self.env.robot_interface._arm_state.tcp_pose_base.position.z,
                self.env.robot_interface._arm_state.tcp_pose_base.orientation.x,
                self.env.robot_interface._arm_state.tcp_pose_base.orientation.y,
                self.env.robot_interface._arm_state.tcp_pose_base.orientation.z,
                self.env.robot_interface._arm_state.tcp_pose_base.orientation.w,
            ])

            curr_wrench = np.array([
                self.env.robot_interface._arm_state.tcp_wrench_ee.force.x,
                self.env.robot_interface._arm_state.tcp_wrench_ee.force.y,
                self.env.robot_interface._arm_state.tcp_wrench_ee.force.z,
                self.env.robot_interface._arm_state.tcp_wrench_ee.torque.x,
                self.env.robot_interface._arm_state.tcp_wrench_ee.torque.y,
                self.env.robot_interface._arm_state.tcp_wrench_ee.torque.z,
            ])
            print("curr_wrench", curr_wrench)
            curr_time = count

            action = np.copy(action_to_take)
            current_pose_in_rot_matrix = self.env._quat_pose_to_rotation_matrix(curr_pose)
            new_rot_mat = action[3:].reshape(3,3)
            action[:3] = np.clip(action[:3], -0.001, 0.001)
            new_pose_in_rot_matrix = np.hstack((current_pose_in_rot_matrix[:3] + action[:3], new_rot_mat.ravel()))
            action_pose = self.env._rotation_matrix_pose_to_quat(new_pose_in_rot_matrix)        

            ########################    
            # execute the action
            ########################
            obs, reward, done, _ = self.env.step(action_to_take)
            curr_state = np.hstack((np.copy(obs[:3]),
                                   np.copy(obs[3:6]),
                                   np.copy(obs[6:15])))
            print(" curr state after executing action", curr_state[:3])
            true_state.append(curr_state)
            print("finished step ", step, ", reward: ", self.env._cost(curr_state, self.goal_state))
            
            ################################################    
            # saving collected data in .h5 file
            ################################################ 
            if step <= self.max_step and not done:
                print("\n... Recording data ... \n ")
                self.data_collector.record(curr_pose, action_pose, curr_wrench, curr_time)
            else:
                
                self.data_collector.flush()
                print(f"... Saving to directory {self.data_collector.directory} ... \n")
            
            step += 1
        
        
        print("Time for 1 rollout: {:0.2f} s\n\n".format(time.time() - rollout_start))
        max_input = np.asarray(max_input)
        print("max input", np.amax(max_input, axis=0))
        print("min input", np.amin(max_input, axis=0))

        ################################################    
        # visualising the executed 2D trajectory
        ################################################ 
        pred_states = np.asarray(predicted_state)
        # norm_pred_states = np.asarray(norm_predicted_state)
        actual_state = np.asarray(true_state)
        x_axis = [index for index in range (len(actual_state))]
        goal_x = np.full((len(actual_state),), 0.269) # hard 0.268 # easy 0.400
        goal_y = np.full((len(actual_state),), -0.412) # hard -0.411 # easy 0.376
        goal_z = np.full((len(actual_state),), 0.182) # hard 0.183 # easy 0.285
        y1_z_axis = actual_state[:,2]
        y2_z_axis = pred_states[:,2]
        y3_x_axis = actual_state[:,0]
        y4_x_axis = pred_states[:,0]
        y5_y_axis = actual_state[:,1]
        y6_y_axis = pred_states[:,1]

        plt.subplot(1,3,1)
        plt.plot(x_axis, y3_x_axis, marker="o", color="green")
        plt.plot(x_axis, y4_x_axis, marker="o", color="red")
        plt.plot(x_axis, goal_x,marker="o", color="blue")
        # plt.plot(x_axis, norm_pred_states[:,0], marker="o", color="yellow")
        plt.xlabel('iteration')
        plt.ylabel('x axis position')
        plt.title('state deviation x axis')

        plt.subplot(1,3,2)
        plt.plot(x_axis, y5_y_axis, marker="o", color="green")
        plt.plot(x_axis, y6_y_axis, marker="o", color="red")
        plt.plot(x_axis, goal_y, marker="o", color="blue")
        # plt.plot(x_axis, norm_pred_states[:,1], marker="o", color="yellow")
        plt.xlabel('iteration')
        plt.ylabel('y axis position')
        plt.title('state deviation y axis')

        plt.subplot(1,3,3)
        plt.plot(x_axis, y1_z_axis, marker="o", color="green")
        plt.plot(x_axis, y2_z_axis, marker="o", color="red")
        plt.plot(x_axis, goal_z, marker="o", color="blue")
        # plt.plot(x_axis, norm_pred_states[:,2], marker="o", color="yellow")
        plt.xlabel('iteration')
        plt.ylabel('z axis position')
        plt.title('state deviation z axis')
        plt.show()

        ################################################    
        # visualising the executed 3D trajectory
        ################################################
        def _vis_trajectory(pred, state):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            max_size = 10
            size_ls = np.arange(0, max_size, max_size / len(state[1:, 1]))

            ax.scatter3D(state[:, 1],
                        state[:, 0],
                        state[:, 2],
                        label='state trajectory',
                        s=size_ls,
                        c='tab:blue')
            ax.scatter3D(pred[:, 1],
                        pred[:, 0],
                        pred[:, 2],
                        label='predicted trajectory',
                        s=size_ls,
                        c='tab:orange')

            ax.set_xlabel('Y')
            ax.set_ylabel('X')
            ax.set_zlabel('Z')
            ax.legend()

            plt.tight_layout()
            plt.show()
        
        _vis_trajectory(pred_states, actual_state)