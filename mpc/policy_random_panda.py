# flake8: noqa
import numpy as np
from scipy.spatial.transform import Rotation as R

class Policy_Random(object):
    """
    Generate one single random action
    """

    def __init__(self, env):
        self.env = env

        # Action space is represented by a tuple of (low, high),
        # which are two numpy vectors that specify the min/max action limits per dimension.
        self.low_value = self.env._action_space.low
        self.high_value = self.env._action_space.high
        self.counter = 0
        self.rand_action = None

    def get_action(self, curr_state, random_sampling_params, hold_action_overrideToOne=False):
        # params for random sampling
        angle_min = random_sampling_params["angle_min"]
        angle_max = random_sampling_params["angle_max"]
        hold_action = random_sampling_params["hold_action"]

        if hold_action_overrideToOne:
            hold_action = 1

        ############################
        # sample set point position
        ############################
        if (self.counter % hold_action)==0:
            # self.rand_set_point = np.random.uniform(self.low_value[:3], self.high_value[:3], 3)
            self.rand_set_point = np.random.normal(0, 2/3, 3)
            # print("random pos:", self.rand_set_point)

            self.rand_force = np.zeros(3)

            self.rand_euler = R.from_euler('zyx', np.random.uniform(angle_min, angle_max, size=3) * np.pi)
            self.rand_rot = self.rand_euler.as_matrix().flatten()
            # self.rand_rot  = np.random.uniform(-1, 1, 9)

            self.rand_action = np.hstack((self.rand_set_point,
                                          self.rand_force,
                                          self.rand_rot))

        # TODO check if action has to turned in to array with numpy
        action = self.rand_action
        # print("random action", action)
        self.counter += 1

        return action