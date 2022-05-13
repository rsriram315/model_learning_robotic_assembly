# flake8: noqa
import time
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from mpc.random_shooting_panda import RandomShooting
from mpc.mppi_panda import MPPI
import matplotlib.pyplot as plt
from utils.data_collection_node import DataCollection
from utils.utils import ensure_dir
from datetime import datetime

class MPCRollout:
    def __init__(self, env, dyn_model, rand_policy, goal_state, params, artifact_cb, param_cb, image_cb):
        self.env = env
        self.dyn_model = dyn_model
        self.max_step = params.max_step
        self.controller_type = params.controller_type
        self.goal_state = goal_state
        self.params = params
        self.artifact_cb = artifact_cb
        self.param_cb = param_cb
        self.image_cb = image_cb

        # init controller parameters
        if params.controller_type == 'random_shooting':
            self.controller = RandomShooting(self.env,
                                             self.dyn_model,
                                             self.cost,
                                             rand_policy, params)
        elif params.controller_type == 'mppi':
            self.params.mppi_gamma = 5
            self.params.mppi_mag_noise = 1
            self.params.mppi_beta = 0.9
            # log params to mlflow
            self.param_cb(dict(mppi_gamma=params.mppi_gamma,
                               mppi_magnitude_noise=params.mppi_mag_noise,
                               mppi_beta=params.mppi_beta))
            self.controller = MPPI(self.env,
                                   self.dyn_model,
                                   self.cost,
                                   rand_policy, params)
    
    def cost(self, curr_state, goal_state ):
        """
        cost function to evaluate sampled trajectories
        Cost function is defined in the environment
        Here cost is not considered as -reward 
        """
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
        true_state.append(curr_state)
        predicted_state.append(curr_state)

        # Init Data colelction node
        if self.params.record_mpc_rollouts:
            now = datetime.now()
            self.data_dir = os.path.join(self.params.rollout_save_dir, self.params.task_type, self.params.controller_type,
                         f"{now.year}_{now.month:02}_{now.day:02}")
            self.param_cb(dict(rollout_save_dir=self.data_dir))
            ensure_dir(self.data_dir)
            self.data_collector = DataCollection(self.env, self.data_dir, self.artifact_cb)
            self.data_collector.reset()

        while not(done or step > self.max_step):
            count += 1
            # get optimal action
            if self.controller_type == 'mppi' and count==1:
                self.controller.mppi_mean = np.tile(curr_state, (self.controller.horizon, 1))
                self.controller.mppi_mean = self.dyn_model.norm.normalize(self.controller.mppi_mean[None,:,:], is_action=True,axis=0)
                self.controller.mppi_mean[:, 3:6] = [0, 0, 0]  # force action should be zero
            
            best_action, pred_next_state = self.controller.get_action(curr_state, self.goal_state, step)
            predicted_state.append(pred_next_state)
            best_action_pos = best_action[:3] - np.copy(self.env._get_obs()[:3])
            best_action_rot = best_action[6:]
            
            action_to_take = np.hstack((best_action_pos,
                                       best_action_rot))
            
            ########################################################################    
            # recording current tcp state and action sent to cartesisan impedence setpoint
            ########################################################################
            curr_state_pose = np.array([
                self.env.robot_interface._arm_state.tcp_pose_base.position.x,
                self.env.robot_interface._arm_state.tcp_pose_base.position.y,
                self.env.robot_interface._arm_state.tcp_pose_base.position.z,
                self.env.robot_interface._arm_state.tcp_pose_base.orientation.x,
                self.env.robot_interface._arm_state.tcp_pose_base.orientation.y,
                self.env.robot_interface._arm_state.tcp_pose_base.orientation.z,
                self.env.robot_interface._arm_state.tcp_pose_base.orientation.w,
            ])

            curr_state_wrench = np.array([
                self.env.robot_interface._arm_state.tcp_wrench_ee.force.x,
                self.env.robot_interface._arm_state.tcp_wrench_ee.force.y,
                self.env.robot_interface._arm_state.tcp_wrench_ee.force.z,
                self.env.robot_interface._arm_state.tcp_wrench_ee.torque.x,
                self.env.robot_interface._arm_state.tcp_wrench_ee.torque.y,
                self.env.robot_interface._arm_state.tcp_wrench_ee.torque.z,
            ])
            curr_time = time.time() - rollout_start

            action = np.copy(action_to_take)
            current_pose_in_rot_matrix = self.env._quat_pose_to_rotation_matrix(curr_state_pose)
            new_rot_mat = action[3:].reshape(3,3)
            action[:3] = np.clip(action[:3], -0.001, 0.001)
            new_pose_in_rot_matrix = np.hstack((current_pose_in_rot_matrix[:3] + action[:3], new_rot_mat.ravel()))
            curr_action_pose = self.env._rotation_matrix_pose_to_quat(new_pose_in_rot_matrix)        

            ########################    
            # execute the action
            ########################
            obs, reward, done, _ = self.env.step(action_to_take)
            curr_state = np.hstack((np.copy(obs[:3]),
                                   np.copy(obs[3:6]),
                                   np.copy(obs[6:15])))
            true_state.append(curr_state)
            print("finished step ", step, ", cost: ", self.env._cost(curr_state, self.goal_state))
            
            ################################################    
            # saving collected data in .h5 file
            ################################################ 
            if self.params.record_mpc_rollouts:
                if step < self.max_step:
                    print("\n... Recording data ... \n ")
                    self.data_collector.record(curr_state_pose, curr_action_pose, curr_state_wrench, curr_time, curr_state, pred_next_state, reward)
                    if done:
                        self.param_cb(dict(success=True,
                                    steps_to_solve_task=step,
                                    rollout_time=time.time()- rollout_start))
                        self.data_collector.flush()
                        print(f"... Saving to directory {self.data_collector.directory} ... \n")
                        break
                else:
                    self.param_cb(dict(success=False,
                                    rollout_time=time.time()- rollout_start))
                    self.data_collector.flush()
                    print(f"... Saving to directory {self.data_collector.directory} ... \n")
            
            step += 1
        
        
        print("Time for 1 rollout: {:0.2f} s\n\n".format(time.time() - rollout_start))

        ################################################    
        # visualising the executed 2D trajectory after rollout
        ################################################ 
        plt.rcParams.update({'font.size': 16})
        pred_states = np.asarray(predicted_state)
        # norm_pred_states = np.asarray(norm_predicted_state)
        actual_state = np.asarray(true_state)
        x_axis = [index for index in range (len(actual_state))]
        goal_x = np.full((len(actual_state),), 0.269) # hard  0.269 # easy 0.400 # reach 0.386
        goal_y = np.full((len(actual_state),), -0.412) # hard -0.412 # easy 0.376 # reach -0.008
        goal_z = np.full((len(actual_state),), 0.1825) # hard 0.1825 # easy 0.285 # reach 0.125
        
        # plt.subplots_adjust(left=0.07, bottom=0.08, right=0.97, top=0.90, wspace=0.25)
        fig, axs  = plt.subplots(1, 3, figsize=(20, 10), constrained_layout=True)
        axs[0].plot(x_axis, actual_state[:,0], marker=".", color="tab:green", label="actual")
        axs[0].plot(x_axis, pred_states[:,0], marker=".", color="tab:red", label="predicted")
        axs[0].plot(x_axis, goal_x,marker=".", color="tab:blue", label="goal")
        axs[0].set_xlabel('Time steps', fontsize=24)
        axs[0].set_ylabel('x-axis position [mm]', fontsize=24)
        axs[0].legend()
        axs[0].grid(linestyle='dotted')

        # plt.subplot(1,3,2)
        axs[1].plot(x_axis, actual_state[:,1], marker=".", color="tab:green", label="actual")
        axs[1].plot(x_axis, pred_states[:,1], marker=".", color="tab:red", label="predicted")
        axs[1].plot(x_axis, goal_y, marker=".", color="tab:blue", label="goal")
        # axs[1].plot(x_axis, norm_pred_states[:,1], marker="o", color="yellow")
        axs[1].set_xlabel('Time steps', fontsize=24)
        axs[1].set_ylabel('y-axis position [mm]', fontsize=24)
        axs[1].legend()
        axs[1].grid(linestyle='dotted')
        
        # plt.subplot(1,3,3)
        axs[2].plot(x_axis, actual_state[:,2], marker=".", color="tab:green", label="actual")
        axs[2].plot(x_axis, pred_states[:,2], marker=".", color="tab:red", label="predicted")
        axs[2].plot(x_axis, goal_z, marker=".", color="tab:blue", label="goal")
        axs[2].set_xlabel('Time steps', fontsize=24)
        axs[2].set_ylabel('z-axis position [mm]', fontsize=24)
        axs[2].legend()
        axs[2].grid(linestyle='dotted')
        if self.params.controller_type == 'random_shooting':
            fig.suptitle('Random shooting rollout: End-effector Trajectory', fontsize=24)
        elif self.params.controller_type == 'mppi':
            fig.suptitle('MPPI rollout: End-effector trajectory', fontsize=24)
        # logging the trajectory to mlflow
        if self.image_cb:
            self.image_cb(fig)
        plt.show()
