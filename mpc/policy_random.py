# flake8: noqa
import numpy as np


class Policy_Random(object):
    def __init__(self, env):
        self.env = env
        self.low_val = -1 * np.ones(self.env.action_space.low.shape)
        self.high_val = np.ones(self.env.action_space.high.shape)
        self.shape = self.env.action_space.shape
        self.counter = 0

        self.rand_action = np.random.uniform(self.low_val, self.high_val,
                                             self.shape)

    def get_action(self, observation, prev_action, random_sampling_params, hold_action_overrideToOne=False):

        # params for random sampling
        sample_pos = random_sampling_params["sample_pos"]
        pos_min = random_sampling_params["pos_min"]
        pos_max = random_sampling_params["pos_max"]
        hold_action = random_sampling_params["hold_action"]

        if hold_action_overrideToOne:
            hold_action = 1

        if sample_pos:
            if prev_action is None:
                # generate action for right now
                self.rand_action = np.random.uniform(self.low_val, self.high_val,
                                                    self.shape)
                action = self.rand_action

                # generate set point position, to be used if next steps might hold_action
                self.pos_sample = np.random.uniform(pos_min, pos_max, self.action_space.low.shape)
                self.direction_num = np.random.randint(0, 2, self.env.action_space.low.space)
                self.pos_sample[self.direction_num == 0] = -self.pos_sample[self.direction_num == 0]
            else:
                if (self.counter % hold_action) == 0:
                    self.pos_sample = np.random.uniform(pos_min, pos_max, self.env.action_space.low.shape)
                    self.direction_num = np.random.randint(0, 2, self.env.action_space.low.space)
                    self.pos_sample[self.direction_num == 0] = -self.pos_sample[self.direction_num == 0]

                    # go opposite direction if you hit limit
                    self.pos_sample[prev_action<=self.low_val] = np.abs(self.pos_sample)[prev_action<=self.low_val]
                    self.pos_sample[prev_action>=self.high_val] = -np.abs(self.pos_sample)[prev_action>=self.high_val]
                # new action
                action = prev_action + self.pos_sample

        self.counter += 1
        return action, 0