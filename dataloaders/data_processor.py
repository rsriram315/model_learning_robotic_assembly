import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d


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


# contact segmentation
class SegmentContact:
    def __init__(self):
        pass

    def contact_time(self, force_xyz, time,
                     subsample=2,
                     novelty_thres=0.15,
                     energy_thres=0.15):
        """
        detecting when the robot contacts the environment

        Params:
            force_xyz: force of x y z axis
        Return:
            start and end index in time
        """
        # subsampling the force as a way to filter out the noise in novelty
        force_xyz = force_xyz[::subsample]
        time = time[::subsample]

        novelty, energy = self._novelty_energy(force_xyz, time)

        # start point should be small, and have large positive novelty
        start_mask = (novelty >= novelty_thres) & (energy <= energy_thres)
        start_candidate = np.where(start_mask)[0]

        # end point should also be small, and have large negative novelty
        end_mask = (novelty <= -novelty_thres) & (energy <= energy_thres)
        end_candidate = np.where(end_mask)[0]

        # if the last energy is not small
        # it could be also an end point (sudden stop of recording)
        if energy[-1] >= energy_thres:
            end_candidate = np.append(end_candidate, time.size - 1)

        # suppress noisy detected boundaries
        start = self._find_start_bounds(start_candidate)
        end = self._find_end_bounds(end_candidate)
        start, end = self._match_start_end(start, end, subsample)

        # multiply the subsampling factor back for indexing
        return start, end

    def _novelty_energy(self, force_xyz, time):
        # use start phase force as baseline
        force_xyz = self._calibrate_force(force_xyz)

        # use the force magnitude sum
        force = np.linalg.norm(force_xyz, axis=1)

        # low-pass filter for forces, otherwise would be too noisy
        force_lp = gaussian_filter1d(force, sigma=8)

        # energy = force_lp**2  # power 2 to suppress small values
        energy = force_lp
        energy /= np.amax(energy)  # normalize energy

        # discrete dervative
        energy_diff = np.diff(energy)
        time_diff = np.diff(time)  # time difference

        novelty = energy_diff / time_diff
        novelty = np.append(novelty, 0.0)
        novelty /= np.amax(abs(novelty))  # normalize to [-1, 1]
        return novelty, energy

    def _calibrate_force(self, force_xyz, start_period=10):
        """
        use start phase force as baseline for contact detection

        force_xyz has shape (num_data, 3)
        """
        offset_x = np.mean(force_xyz[:start_period, 0])
        offset_y = np.mean(force_xyz[:start_period, 1])
        offset_z = np.mean(force_xyz[:start_period, 2])

        force_xyz[:, 0] -= offset_x
        force_xyz[:, 1] -= offset_y
        force_xyz[:, 2] -= offset_z

        return force_xyz

    def _find_start_bounds(self, candidate, tolerant=10):
        """
        find start boundaries, keep the first found bound
        """
        bounds = [candidate[0]]
        bound_prev = bounds[0]

        for idx in range(candidate.size):
            bound_new = candidate[idx]

            if bound_new - bound_prev >= tolerant:
                bounds.append(bound_new)

            bound_prev = bound_new
        return np.array(bounds)

    def _find_end_bounds(self, candidate, tolerant=10):
        """
        find end boundary, keep the last fined bound
        """
        bounds = [candidate[0]]

        for idx in range(candidate.size):
            bound_new = candidate[idx]

            if bound_new - bounds[-1] >= tolerant:
                bounds.append(bound_new)
            else:
                bounds[-1] = bound_new
        return np.array(bounds)

    def _match_start_end(self, start, end, subsample):
        """
        Assume only one contacting phase exits and it is continuous
        match the first found start and the last found end
        """
        return [start[0] * subsample], [end[-1] * subsample]
