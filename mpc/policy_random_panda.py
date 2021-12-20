# flake8: noqa
import numpy as np
from numpy.core.fromnumeric import size
from scipy.spatial.transform import Rotation as R

class Policy_Random(object):
    """
    Generate one single random action
    """

    def __init__(self, env):
        self.env = env

        # Action space is represented by a tuple of (low, high),
        # which are two numpy vectors that specify the min/max action limits per dimension.
        self.counter = 1
        self.rand_action = None

    def get_action(self, curr_state, random_sampling_params, hold_action_overrideToOne=False, traj_count=0):
        # params for random sampling
        angle_min = random_sampling_params["angle_min"]
        angle_max = random_sampling_params["angle_max"]
        hold_action = random_sampling_params["hold_action"]

        if hold_action_overrideToOne:
            hold_action = 1
        
        if self.counter == 1:
            self.init_rot = R.from_matrix(curr_state[6:15].reshape(3,3))
            self.init_euler_angle = self.init_rot.as_euler('zyx')
            print("self.init_euler_angle", self.init_euler_angle)

        ############################
        # sample set point position
        ############################
        if (self.counter % hold_action)==0:
            # toy actions for checking the model
            z_set_point_ls = np.arange(0.45, -0., -0.002)
            # # self.rand_set_point = np.array([0.307, 0., z_set_point_ls[traj_count]])
            
            # sample actions in [-1,1]
            self.rand_set_point = np.random.uniform(np.array([-1, -1, -1]), np.array([1, 1, 1]), 3)
            # self.rand_set_point = np.array([0.69785281, -0.01056377, np.random.uniform(-1, 1)])
            self.rand_force = np.zeros(3)
            # self.rand_rot = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
            self.rand_euler = R.from_euler('zyx', np.random.uniform(angle_min, angle_max, size=3) * np.pi)
            self.rand_rot = self.rand_euler.as_matrix().flatten()

            self.rand_action = np.hstack((self.rand_set_point,
                                          self.rand_force,
                                          self.rand_rot))

        action = self.rand_action

        self.counter += 1

        return action