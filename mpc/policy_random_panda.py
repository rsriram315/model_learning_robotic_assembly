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
            self.init_rot = R.from_matrix(curr_state[3:12].reshape(3,3))
            self.init_euler_angle = self.init_rot.as_euler('zyx')
            # print("self.init_quat", self.init_rot.as_quat())
            # print("self.init_euler_angle", self.init_rot.as_euler('zyx', degrees=True))
            # print("self.init_rot_mat", self.init_rot.as_matrix().reshape(3,3))

        ############################
        # sample set point position
        ############################
        if (self.counter % hold_action)==0:            
            # sample actions in [-1,1]
            self.rand_set_point = np.random.uniform(np.array([-1, -1, -1]), np.array([1, 1, 1]), 3)
            self.euler_delta = np.random.uniform(angle_min, angle_max, size=3) * np.pi
            self.rand_euler = R.from_euler('zyx', (self.init_euler_angle + self.euler_delta) ,degrees=False)
            # print("self.init_euler_angle", self.init_rot.as_euler('zyx', degrees=True))
            # print("self.rand_delta",  (self.euler_delta * 180)/np.pi)
            # print("self.rand_euler", self.init_rot.as_euler('zyx', degrees=True) + (self.euler_delta * 180)/np.pi)
            # print("self.rand_force",self.rand_force)
            # print("self.rand_set_point", self.rand_set_point)
            # print("self.rand_quat", self.rand_euler.as_quat())
            # print("self.rand_mat", self.rand_euler.as_matrix().reshape(3,3))
            self.rand_rot = self.rand_euler.as_matrix().flatten()

            self.rand_action = np.hstack((self.rand_set_point,
                                          self.rand_rot))

        action = self.rand_action

        self.counter += 1

        return action