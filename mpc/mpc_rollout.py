# flake8: noqa
import numpy as np
import time

from mpc import RandomShooting


class MPCRollout:
    def __init__(self, env, dyn_model, rand_policy, params):
        self.env = env
        self.dyn_model = dyn_model
        self.rand_policy = rand_policy
        self.rollout_length = params.rollout_length
        self.K = params.K

        self.reward_fn = env.unwrapped_env.get_reward

        # init controller
        self.controller_randshooting = RandomShooting(self.env, self.dyn_model, self.reward_fn, rand_policy, params)

    def perform_rollout(self, starting_fullenvstate, starting_observation, controller_type):
        """
        Args:
            starting_fullenvstate: full state of the mujoco env (enough to allow resetting it)
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

        # select controller type
        if controller_type == 'rand':
            get_action = self.controller_randshooting.get_action

        # list of saving
        traj_taken = []
        traj_taken_K = []
        actions_taken = []
        actions_taken_K = []
        rewards = []
        scores = []
        env_infos = []
        list_mpe_1step = []

        # init vars for rollout
        total_reward_for_episode = 0
        step = 0
        self.starting_fullenvstate = starting_fullenvstate

        # initialize first K states/actions
        zero_action = np.zeros((self.env.action_dim, ))
        curr_state = np.copy(starting_observation)
        curr_state_K = [curr_state]  # not K yet, but will be

        # take (K - 1) steps of action 0
        for _ in range(self.K - 1):
            # take step of action 0
            curr_state, reward, _, env_info = self.env.step(zero_action)
            step += 1

            actions_taken.append(zero_action)
            curr_state_K.append(curr_state)

            # save info
            rewards.append(reward)
            scores.append(env_info["score"])
            env_infos.appen(env_info)
            total_reward_for_episode += reward

            # rewards/actions/etc. are populated during these first K steps
            # but traj_taken_K / traj_taken are not because curr_state_K is not of size K yet

        traj_taken.append(curr_state)
        traj_taken_K.append(curr_state_K)

        # loop over steps in rollout
        done = False
        while not(done or step >= self.rollout_length):
            # get optimal action
            # curr_state_K: past K states
            # actions_taken: past all actions (taken so far in this rollout)
            best_action, predicted_states_ls = get_action(curr_state_K, actions_taken)

            action_to_take = np.copy(best_action)
            clean_action = np.copy(action_to_take)
            action_to_document = np.copy(clean_action)

        # execute the action
        next_state, reward, done, env_info = self.env.step(action_to_take)

        # get predicted next_state, use it to calculate model prediction error (mpe)
        # get updated mean/std from the dynamics model
        # TODO change the way of normalization
        curr_mean_x = self.dyn_models.normalization_data.mean_x
        curr_std_x = self.dyn_models.normalization_data.std_x
        next_state_preprocessed = (next_state - curr_mean_x) / curr_std_x

        # the most recent (K-1) actions
        actions_Kmin1 = np.array(actions_taken[-(self.K - 1):])  # [K-1, action_dim]

        # create
