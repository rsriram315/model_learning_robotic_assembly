# flake8: noqa
import time
import numpy as np
import robosuite.utils.transform_utils as T

from mpc.random_shooting import RandomShooting

FORCE_SCALING_FACTOR = 0.000001


class MPCRollout:
    def __init__(self, env, dyn_model, random_policy, params):
        self.env = env
        self.dyn_model = dyn_model
        self.random_policy = random_policy
        self.rollout_length = params.rollout_length
        self.K = params.K

        # self.reward_fn = env.reward
        self.cost_fn = self.cost

        # init controller
        self.controller_randshooting = RandomShooting(self.env,
                                                      self.dyn_model,
                                                      self.cost_fn,
                                                      random_policy, params)
        self.controller_mppi = None

    def cost(self, curr_state, goal_state):
        # reward is normalized
        mpc_cost = 1 - self.env.reward(curr_state=curr_state, goal_state=goal_state)
        return mpc_cost

    def perform_rollout(self, starting_envstate, starting_observation, controller_type):
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
        ########################
        # select controller type
        ########################
        if controller_type == 'rand':
            get_action = self.controller_randshooting.get_action
        elif controller_type == 'mppi':
            get_action = self.controller_mppi.get_action

        ##################
        # lists for saving
        ##################
        traj_taken = []
        actions_taken = []
        rewards = []
        env_infos = []

        #######################
        # init vars for rollout
        #######################
        total_reward_for_episode = 0
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
        while not(done or step >= self.rollout_length):
            # get optimal action
            best_action = self.controller_randshooting.get_action(curr_state)

            ################################################
            # transform the action in base frame to ee frame
            ################################################
            # make T_in_B as homogeneous matrix, and minus the curr_state pose
            target_in_B = T.make_pose(best_action[:3], best_action[6:15].reshape((3, 3)))
            # The tool center point frame expressed in the base frame
            curr_pose_in_B = self.env.unwrapped.robots[0]._hand_pose

            T_from_B_to_EE = T.pose_inv(curr_pose_in_B)
            target_in_EE =  T.pose_in_A_to_pose_in_B(target_in_B, T_from_B_to_EE)

            action_pos = target_in_EE[:3, -1]

            action_orn = T.mat2euler(target_in_EE[:3, :3])

            action_gripper = [-1]
            action_to_take = np.hstack((action_pos,
                                        action_orn,
                                        action_gripper))

            ########################
            # execute the action
            ########################
            _, reward, done, env_info = self.env.step(action_to_take)

            self.env.render()

            rewards.append(reward)
            env_infos.append(env_info)
            actions_taken.append(action_to_take)
            total_reward_for_episode += reward

            curr_state = np.hstack(
                (self.env.unwrapped.robots[0]._hand_pos,
                 self.env.unwrapped.robots[0].ee_force,
                 self.env.unwrapped.robots[0]._hand_orn.flatten()))

            traj_taken.append(curr_state)

            print("done step ", step, ", reward: ", reward)

            step += 1

            rollout_info = dict(
                starting_state=starting_envstate,
                observations=np.array(traj_taken),
                actions=np.array(actions_taken),

                rollout_rewardsPerStep=np.array(rewards),
                rollout_rewardTotal=total_reward_for_episode,

                env_infos=env_infos,
            )
        print("Time for 1 rollout: {:0.2f} s\n\n".format(time.time() - rollout_start))
        return rollout_info
