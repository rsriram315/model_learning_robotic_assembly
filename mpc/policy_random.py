# flake8: noqa
import numpy as np


class Policy_Random(object):
    """
    Generate one single random action
    """

    def __init__(self, env):
        self.env = env

        # Action space is represented by a tuple of (low, high),
        # which are two numpy vectors that specify the min/max action limits per dimension.
        self.low_val, self.high_val = self.env.action_spec
        self.shape = self.env.action_dim
        self.counter = 0

        self.rand_action = np.random.uniform(self.low_val, self.high_val,
                                             self.shape)

    def get_action(self, curr_state, random_sampling_params, hold_action_overrideToOne=False):

        # params for random sampling
        sample_rot = random_sampling_params["sample_rot"]
        angle_min = random_sampling_params["angle_min"]
        angle_max = random_sampling_params["angle_max"]
        hold_action = random_sampling_params["hold_action"]

        if hold_action_overrideToOne:
            hold_action = 1

        ############################
        # sample set point position
        ############################
        if not sample_rot:
            if (self.counter % hold_action)==0:
                # only sample set point position
                z = np.random.uniform(self.low_val[2], self.high_val[2], 1)
                self.rand_set_point = np.hstack((curr_state[:2], z))
                self.rand_force = np.zeros(3)
                self.rand_rot = np.copy(curr_state[6:15])
                self.rand_action = np.hstack((self.rand_set_point,
                                              self.rand_force,
                                              self.rand_rot))
            action = self.rand_action

        self.counter += 1

        return action
