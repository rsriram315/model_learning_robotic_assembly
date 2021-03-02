import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R


def data_stats(data):
    """
    mean and variance
    """
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    return data_mean, data_std


def normalize(data, mean, std):
    """
    standard normalization for data
    """
    # https://en.wikipedia.org/wiki/Normalization_(statistics)
    # normalized_data = (data - mean) / std

    # Subtract the mean, and scale to the interval [-1,1]
    data_minusmean = data - mean
    normalized_data = data_minusmean / np.max(np.abs(data_minusmean))
    return normalized_data


class Interpolation:
    def __init__(self, data, time, interpolation):
        self.data = data
        self.time = time
        self.interpolation = interpolation

        if self.interpolation == "cubic_spline":
            self.cs_ls = self._get_cubic_spline_fn()
        elif self.interpolation == "slerp":
            self.rot = R.from_quat(self.data)
            self.slerp_fn = self._get_slerp_fn()

    def interp(self, time_stamp):
        if self.interpolation == "zero_order":
            return self._zero_order_interp(time_stamp)
        elif self.interpolation == "cubic_spline":
            return self._cubic_spline_interp(time_stamp)
        elif self.interpolation == "slerp":
            return self._slerp(time_stamp)

    def _zero_order_interp(self, time_stamp):
        """
        zero order interpolation, we choose the nearest neighbor data
        BEFORE the sampling time stamp.
        """
        res = []

        for t in time_stamp:
            mask = list(map(int, (self.time - t) <= 0))
            idx = sum(mask) - 1
            data = self.data[idx]

            res.append(data)
        return np.array(res)

    def _cubic_spline_interp(self, time_stamp):
        # cubic spline for each dimension indivually
        res = np.array([cs(time_stamp) for cs in self.cs_ls])
        # stiching each dimension
        cs = np.array([res[:, i] for i in range(res.shape[1])])
        return cs

    def _get_cubic_spline_fn(self):
        num_fns = self.data.shape[1]

        cs_ls = []
        for i in range(num_fns):
            cs_ls.append(CubicSpline(self.time, self.data[:, i]))
        return cs_ls

    def _slerp(self, time_stamp):
        # return self.slerp_fn(time_stamp).as_quat()
        # output the interpolated euler vector
        return self.slerp_fn(time_stamp).as_rotvec()

    def _get_slerp_fn(self):
        return Slerp(self.time, self.rot)

    def _normalize(self, vec):
        """
        normalize a vector to become unit length
        """
        return (vec.T / np.sqrt(np.sum(vec**2, axis=-1))).T
