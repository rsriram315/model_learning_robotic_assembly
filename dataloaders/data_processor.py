import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d


_FLOAT_EPS = np.finfo(np.float64).eps


def add_noise(data, noise_lv=0.1):
    noise = np.random.normal(0, noise_lv, size=3)
    data[3:6] = data[3:6] + noise
    return data


def homogeneous_transform(data, r_noise=0.001, t_noise=0.1):
    # generate random homegeneous_matrix
    noisy_r = R.from_euler('zyx', np.random.normal(0, r_noise, size=3) * np.pi)
    rotation_matrix = noisy_r.as_matrix()
    translation = np.random.normal(0, t_noise, size=3)

    homogeneous_matrix = np.zeros((4, 4))
    homogeneous_matrix[:3, :3] = rotation_matrix
    homogeneous_matrix[:, -1] = np.hstack((translation, 1))
    data[:3] = np.copy(homogeneous_matrix @
                       np.hstack((data[:3], 1)))[:3]
    data[6:] = (np.copy(data[6:].reshape(3, 3) @
                        homogeneous_matrix[:3, :3]).flatten())

    return data


def rotation_diff(target_rot, init_rot):
    init_r = R.from_matrix(init_rot.reshape(-1, 3, 3))
    target_r = R.from_matrix(target_rot.reshape(-1, 3, 3))
    diff_r = target_r * init_r.inv()
    return diff_r.as_matrix().reshape(-1, 9)


def recover_rotation(diff_state, init_state):
    init_rot = R.from_matrix(init_state[:, 6:15].reshape(-1, 3, 3))
    diff_rot = R.from_matrix(diff_state[:, 6:15].reshape(-1, 3, 3))
    target_rot = diff_rot * init_rot

    # output orientation matrix
    recover_target = np.copy(diff_state)
    recover_target[:, 6:15] = target_rot.as_matrix().reshape(-1, 9)
    return recover_target


def add_euler_angle(state):
    rot = R.from_matrix(state[:, 6:].reshape(-1, 3, 3))
    # output euler angles
    euler_angle = rot.as_euler('xyz', degrees=True)
    return np.hstack((state, euler_angle))


class Normalization:
    """
    Normalization of data, (x - min(x)) / (max(x) - min(x)).
    """
    def __init__(self, stats):


        self.stat_1 = stats["stat_1"]
        self.stat_2 = stats["stat_2"]
        self.stat_3 = stats["stat_3"]
        self.stat_4 = stats["stat_4"]

    def _stats(self, data):
        array_min = np.amin(data, axis=0)
        array_range = np.amax(data, axis=0) - array_min + _FLOAT_EPS

        array_min[:, 6:] = np.zeros(9) - 1
        array_range[:, 6:] = np.ones(9) + 1
        return array_min, array_range

    def get_stats(self):
        stats = {"stat_1": self.stat_1,
                 "stat_2": self.stat_2,
                 "stat_3": self.stat_3,
                 "stat_4": self.stat_4}
        return stats

    def normalize(self, data, is_res=False, is_action=False, axis=1):
        """
        standard normalization for data
        """
        if is_res:
            if self.stat_3 is None or self.stat_4 is None:
                array_min, array_range = self._stats(data)
                self.stat_3, self.stat_4 = array_min, array_range
            else:
                array_min, array_range = self.stat_3, self.stat_4
        else:
            if self.stat_1 is None or self.stat_2 is None:
                array_min, array_range = self._stats(data)
                self.stat_1, self.stat_2 = array_min, array_range
            else:
                array_min, array_range = self.stat_1, self.stat_2

        if is_action:
            dim = 1
            range = array_range[1]
            min = array_min[1]
        else:
            # state_actions has dim [N_samples, 2, 15]
            # best action has dim [1, 1, 15]
            dim = data.shape[1]  # check if state action or not
            range = array_range[:dim]
            min = array_min[:dim]  

        normalized_data = (data - min) / range
        if dim == 1:
            normalized_data = np.squeeze(normalized_data, axis=axis)

        return 2 * (normalized_data - 0.5)

    def inv_normalize(self, data, is_res=False, is_action=False):
        if is_res:
            array_min, array_range = self.stat_3, self.stat_4
        else:
            array_min, array_range = self.stat_1, self.stat_2

        if is_action:
            dim = 1
            range = array_range[1]
            min = array_min[1]
        else:
            dim = data.shape[1]  # check if state action or not
            range = array_range[:dim]
            min = array_min[:dim]

        scaled_data = (data / 2 + 0.5) * (range - _FLOAT_EPS)
        inversed_data = scaled_data + min
        if dim == 1:
            inversed_data = np.squeeze(inversed_data, axis=1)
        return inversed_data


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

        cs_ls = []  # list of cubic spline interpolated values
        for i in range(num_fns):
            cs_ls.append(CubicSpline(self.time, self.data[:, i]))
        return cs_ls

    def _slerp(self, time_stamp):
        # output rotation matrix
        # TODO slerp cannot extrapolate, you need to adjust the timestamp range
        return self.slerp_fn(time_stamp).as_matrix().reshape((-1, 9))

    def _get_slerp_fn(self):
        return Slerp(self.time, self.rot)
    

# contact segmentation
class SegmentContact:
    def __init__(self):
        pass

    def contact_time(self,
                     force_xyz,
                     time,
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

        # Assumption 1:
        # start point should be small, and have large positive novelty
        start_mask = (novelty >= novelty_thres) & (energy <= energy_thres)
        start_candidate = np.where(start_mask)[0]

        # Assumption 2:
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
