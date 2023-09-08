# flake8: noqa
import time
import numpy as np
import robosuite.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R
from mpc.random_shooting import RandomShooting
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
        # reward is normalized
        mpc_cost = 1 - self.env.reward(curr_state=curr_state, goal_state=goal_state)
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
            # print("best_action", best_action)
            ####################################################
            # transform the action in base frame to global frame
            ####################################################
            curr_G_pos = np.copy(self.env.unwrapped.sim.data.body_xpos[self.env.peg_body_id])
            # print("curr_pos eef", curr_G_pos)
            # curr_G_orn = np.copy(self.env.unwrapped.sim.data.body_xmat[self.env.peg_body_id])
            # curr_G_pose = T.make_pose(curr_G_pos, curr_G_orn)

            # target_in_G = T.make_pose(best_action[:3], best_action[6:15].reshape((3, 3)))

            # diff_in_G = curr_G_pose.dot(T.pose_inv(target_in_G))

            # action_pos = best_action[:3] - curr_G_pos
            action_pos = np.array([0, 0, -0.001])
            # action_orn = T.mat2euler(best_action[6:15].reshape((3, 3)))
            action_orn = np.array([0, 0, 0])
            action_gripper = [-1]
            action_to_take = np.hstack((action_pos,
                                        action_orn,
                                        action_gripper))
            # print("action to take:",action_to_take)

            ########################
            # execute the action
            ########################
            _, reward, done, _ = self.env.step(action_to_take)
            # print(f"force magnit"ude {np.linalg.norm(self.env.robots[0].ee_force)}\n")
            self.env.render()

            curr_state = np.hstack(
                (self.env.unwrapped.sim.data.body_xpos[self.env.peg_body_id],
                 self.env.unwrapped.robots[0].ee_force,
                 self.env.unwrapped.sim.data.body_xmat[self.env.peg_body_id].flatten()))
            print("obs", curr_state[:3])
            # print("finished step ", step, ", reward: ", reward)
            step += 1

        print("Time for 1 rollout: {:0.2f} s\n\n".format(time.time() - rollout_start))
