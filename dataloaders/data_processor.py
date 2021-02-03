import numpy as np
from math import sqrt, acos
from scipy.interpolate import CubicSpline


def data_stats(data):
    """
    mean and variance
    """
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    return data_mean, data_std


def normalize(data, mean, std):
    """
    standard normalization
    """
    # https://en.wikipedia.org/wiki/Normalization_(statistics)
    # normalized_data = (data - mean) / std

    # Subtract the mean, and scale to the interval [-1,1]
    data_minusmean = data - mean
    normalized_data = data_minusmean / np.max(np.abs(data_minusmean))
    return normalized_data


def quaternion_to_axis_angle(q):
    """
    transform quaternion to axis angel representation

    Params:
        q: quaternion with x, y, z, w
    Return:
        [axis_x, axis_y, axis_z, angle]
    """
    # the quaternion is x, y, z, w
    axis_angle = np.zeros(4)
    axis_angle[3] = 2 * acos(q[3])
    axis_angle[0] = q[0] / sqrt(1.0 - q[3] * q[3])
    axis_angle[1] = q[1] / sqrt(1.0 - q[3] * q[3])
    axis_angle[2] = q[2] / sqrt(1.0 - q[3] * q[3])
    return axis_angle


def axis_angle_to_euler_vector(axis_angle):
    """
    axis angle to euler vector, [theta * v_x, theta * v_y, theta * v_z]
    """
    euler_vector = axis_angle[:3] * axis_angle[3]
    return euler_vector


def euler_vector_to_axis_angle(euler_vector):
    """
    euler vector back to axis angle
    """
    axis_angle = np.zeros(4)
    axis_angle[3] = sqrt(np.sum(np.power(euler_vector, 2)))
    axis_angle[:3] = euler_vector / axis_angle[3]
    return axis_angle


class Interpolation:
    def __init__(self, data_time, interpolation):
        self.data_time = data_time
        self.interpolation = interpolation
        if self.interpolation == "cubic_spline":
            self._get_cubic_spline_fn(self.data_time)

    def interp(self, time_stamp):
        if self.interpolation == "zero_order":
            return self._zero_order_interp(time_stamp)
        elif self.interpolation == "cubic_spline":
            return self._cubic_spline_interp(time_stamp)

    def _zero_order_interp(self, time_stamp):
        """
        not exactly a zero order, we choose the nearest neighbor data
        BEFORE the sampling time stamp.
        """
        mask = list(map(int, (self.data_time["time"] - time_stamp) <= 0))
        idx = sum(mask) - 1
        return self.data_time["data"][idx]

    def _cubic_spline_interp(self, time_stamp):
        # cubic spline
        return [self.cs_pos_x(time_stamp),
                self.cs_pos_y(time_stamp),
                self.cs_pos_z(time_stamp),
                self.cs_ev_x(time_stamp),
                self.cs_ev_y(time_stamp),
                self.cs_ev_z(time_stamp)]

    def _get_cubic_spline_fn(self, data_time):
        self.cs_pos_x = CubicSpline(data_time["time"],
                                    data_time["data"][:, 0])
        self.cs_pos_y = CubicSpline(data_time["time"],
                                    data_time["data"][:, 1])
        self.cs_pos_z = CubicSpline(data_time["time"],
                                    data_time["data"][:, 2])
        self.cs_ev_x = CubicSpline(data_time["time"],
                                   data_time["data"][:, 3])
        self.cs_ev_y = CubicSpline(data_time["time"],
                                   data_time["data"][:, 4])
        self.cs_ev_z = CubicSpline(data_time["time"],
                                   data_time["data"][:, 5])
