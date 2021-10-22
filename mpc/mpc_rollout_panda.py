# flake8: noqa
import time
import numpy as np
import robosuite.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R
from mpc.random_shooting_panda import RandomShooting
from mpc.mppi import MPPI


class MPCRollout:
    def __init__(self, env, dyn_model, rand_policy, goal_state, params):
        self.env = env
        self.dyn_model = dyn_model
        self.max_step = params.max_step
        self.controller_type = params.controller_type

        self.goal_state = goal_state

        # init controller
        if params.controller_type == 'rand_shooting':
            self.controller = RandomShooting(self.env,
                                             self.dyn_model,
                                             self.cost,
                                             rand_policy, params)
        elif params.controller_type == 'mppi':
            params.mppi_gamma = 10
            params.mppi_mag_noise = 0.8
            params.mppi_beta = 0.9
            self.controller = MPPI(self.env,
                                   self.dyn_model,
                                   self.cost,
                                   rand_policy, params)
    
    def cost(self, curr_state, goal_state):
        """
        TODO fix reward function
        """
        # reward is normalized
        mpc_cost = self.env._cost(curr_state=curr_state, goal_state=goal_state)
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
        step = 0
        self.starting_envstate = starting_envstate

        ###################################
        # initialize first K states/actions
        ###################################
        curr_state = np.copy(starting_envstate)

        #######################################
        # loop over steps in rollout
        #######################################
        done = False
        count = 0
        while not(done or step >= self.max_step):
            count += 1
            # get optimal action
            if self.controller_type == 'mppi' and count == 1:
                self.controller.mppi_mean = np.tile(curr_state, (self.controller.horizon, 1))
                self.controller.mppi_mean[:, 3:6] = [0, 0, 0]  # force action should be zero
                self.controller.mppi_mean[:, 6:15] = np.eye(3).flatten()  # action rotation is delta
            best_action = self.controller.get_action(curr_state, self.goal_state)
            # print("curr pos", self.env._get_obs()[:3])
            best_action_pos = best_action[:3] - np.copy(self.env._get_obs()[:3])
            best_action_rot = best_action[6:]
            # best_action_rot = best_action_rot_matrix.as_rotvec()
            # print(best_action_rot)
            action_to_take = np.hstack((best_action_pos,
                                       best_action_rot))
            print("action to take:",action_to_take)
            ########################    
            # execute the action
            ########################
            _, reward, done, _ = self.env.step(action_to_take)

            curr_robot_pos = np.copy(self.env._get_obs()[:3])
            curr_r = R.from_rotvec(np.copy(self.env._get_obs()[3:6]))
            curr_robot_orn = curr_r.as_matrix().flatten()
            curr_robot_force = np.copy(self.env._get_obs()[6:9])

            curr_state = np.hstack((curr_robot_pos,
                                   curr_robot_force,
                                   curr_robot_orn))
            
            print("finished step ", step, ", reward: ", reward)
            step += 1
        
        print("Time for 1 rollout: {:0.2f} s\n\n".format(time.time() - rollout_start))