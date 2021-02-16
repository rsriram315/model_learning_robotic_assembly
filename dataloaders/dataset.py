import numpy as np
import h5py
import torch
from functools import partial
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset, random_split
from .data_processor import data_stats, normalize, Interpolation


class DemoDataset(Dataset):
    def __init__(self, root, fnames, sample_freq, state_attrs, action_attrs,
                 preprocess, contact_only):
        """
        form (state, action) pair as x, and state at right next time stamp
        as label, pack them together.

        Args:
            params: dataset dict in the config.json
        """
        super().__init__()
        self.GRIPPER_CLOSED_THRESHOLD = 0.013

        self.root = Path(root)
        self.data_paths = [self.root / fn for fn in fnames]

        self.contact_only = contact_only

        self.sample_freq = sample_freq
        self.state_attrs = state_attrs
        self.action_attrs = action_attrs

        self.state_start_times = []
        self.state_end_times = []

        self.action_start_times = []
        self.action_end_times = []

        self.sample_time = []
        self.paired_states_actions = []

        # debug variables
        self.all_actions_time = None
        self.all_actions_pos = None
        self.all_states_force = None
        self.all_states_pos = None
        self.all_states_time = None

        self.preprocess = preprocess

        self._read_all_demos()

        if self.preprocess["normalize"]:
            # state
            self.paired_states_actions[:, 0] = \
                self._normalize_pos(self.paired_states_actions[:, 0].copy())
            # action
            self.paired_states_actions[:, 1] = \
                self._normalize_pos(self.paired_states_actions[:, 1].copy())

    def __len__(self):
        return len(self.paired_states_actions)

    def __getitem__(self, idx):
        state, action = self.paired_states_actions[idx]
        sample = np.concatenate((state, action))
        # simpliest case where action is the target state
        target = action
        return np.float32(sample), np.float32(target)

    def _all_time(self):
        pass

    def _all_data(self):
        pass

    def _read_all_demos(self):
        for data_path in self.data_paths:
            states, actions = self._read_one_demo(data_path, self.state_attrs,
                                                  self.action_attrs,
                                                  self.contact_only)

            # debug variables
            self.all_states_time = states["time"]
            self.all_states_force = states["force"]
            self.all_states_pos = states["pos"]
            self.all_actions_pos = actions["pos"]
            self.all_actions_time = actions["time"]

            self.paired_states_actions.extend(
                self._pair_state_action(self.sample_freq, states, actions))

        self.paired_states_actions = np.array(self.paired_states_actions)

    def _read_one_demo(self,
                       data_path,
                       states,
                       actions,
                       contact_only,
                       time="time_stamp"):
        """
        read data from h5 file
        should I discard the time when the arm is not grasping the object?

        Params:
            data_path: data file path
            sample_freq: sampling frequency
            states: specified states for current task
            actions: specified actions for current task
            time: name of time stamp in the data
        Return:
            tuple of states (state, time), actions (action, time).
            state and action are (position, orientation) here.
        """
        with h5py.File(data_path, 'r') as f:
            # determine when the arm grasp and release object
            gripper_t = np.array(f['franka_gripperjoint_states'][time])
            gripper_s = np.array(f['franka_gripperjoint_states']['q'])
            grasp_start_t, grasp_stop_t = \
                self._grasp_time(gripper_s, gripper_t)

            data = [{}, {}]  # [states, actions]

            for n, i in enumerate([states, actions]):
                # clip time to avoid negative value
                t = np.clip(np.asarray(f[i["name"]][time]), 0, None)

                start_idx = np.argmin(abs(t - grasp_start_t))
                stop_idx = np.argmin(abs(t - grasp_stop_t))

                # discard the data when the arm not grasping the object
                d = []
                for attrs in i["attrs"]:
                    d.append(f[i["name"]][attrs][start_idx:stop_idx])
                t = t[start_idx:stop_idx]
                d = np.hstack(d)

                if contact_only:
                    # extract contact phase
                    is_state = True if n == 0 else False
                    t, d = self._contact(t, d, is_state)

                data[n]["pos"] = np.array(d[:, :3])
                data[n]["rot"] = np.array(d[:, 3:7])
                data[n]["force"] = np.array(d[:, 7:10])
                data[n]["wrench"] = np.array(d[:, 10:13])
                data[n]["time"] = np.array(t)

        return data

    def _grasp_time(self, states, time_stamp):
        """
        detect when the gripper grasp and release the object
        only one start time and one stop time are assumed in this implemetation

        Params:
            states: state of the gripper joint, q
            time_stamp: time stamp of gripper joint states
        Return:
            start_time: the first recorded time when the gripper grasp the
                        object
            stop_time: the last recorded time when the gripper grasp the object
        """
        mask_thres = [s > self.GRIPPER_CLOSED_THRESHOLD for s in states[:, 0]]
        mask_t = np.convolve(list(map(int, mask_thres)), [1, -1], 'valid')

        start_idx = np.argmin(mask_t) + 1
        stop_idx = np.argmax(mask_t)

        start_time = time_stamp[start_idx]
        stop_time = time_stamp[stop_idx]

        return start_time, stop_time

    def _contact(self, t, d, is_state):
        """
        extract the contacting phase of the data and the corresponding time
        """
        t_new = []
        d_new = []

        contact_start = []
        contact_end = []

        if is_state:
            force = d[:, 7:10]  #
            contact_start, contact_end = \
                self._contact_time(force, t)

            self.state_start_times = [t[s] for s in contact_start]
            self.state_end_times = [t[e] for e in contact_end]

        else:  # is_action
            contact_start = [np.argmin(abs(t - s))
                             for s in self.state_start_times]
            contact_end = [np.argmin(abs(t - e))
                           for e in self.state_end_times]

            self.action_start_times = [t[s] for s in contact_start]
            self.action_end_times = [t[e] for e in contact_end]

        for s, e in zip(contact_start, contact_end):
            t_new.append(t[s:e])
            d_new.append(d[s:e, :])

        return np.hstack(t_new), np.vstack(d_new)

    def _contact_time(self, force_xyz, time,
                      novelty_thres=0.1,
                      energy_thres=0.1):
        """
        detecting when the robot contacts the environment

        Params:
            force_xyz: force of x y z axis
        Return:
            start and end index in time
        """
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
        start, end = self._match_start_end(start, end)

        return start, end

    def _pair_state_action(self, sample_freq, states, actions):
        """
        form state action pair from one demo
        """

        start_time = max(min(states["time"]), min(actions["time"]))
        end_time = min(max(states["time"]), max(actions["time"]))

        print(f"dataset start time {start_time}, end time {end_time}")

        sample_time = np.arange(start_time, end_time, 1.0/sample_freq)
        self.sample_time.extend(sample_time)
        print(f"there are {sample_time.shape[0]} samples")

        # position interpolation
        states_pos_interp = Interpolation(states["pos"], states["time"],
                                          self.preprocess["interp"]["pos"])
        actions_pos_interp = Interpolation(actions["pos"], actions["time"],
                                           self.preprocess["interp"]["pos"])

        # rotation interpolation
        states_rot_interp = Interpolation(states["rot"], states["time"],
                                          self.preprocess["interp"]["rot"])
        actions_rot_interp = Interpolation(actions["rot"], actions["time"],
                                           self.preprocess["interp"]["rot"])

        # force interpolation
        states_force_interp = \
            Interpolation(states["force"], states["time"],
                          self.preprocess["interp"]["force"])
        actions_force_interp = \
            Interpolation(actions["force"], actions["time"],
                          self.preprocess["interp"]["force"])

        # concatenate interpolated values
        states_interp = \
            np.hstack((states_pos_interp.interp(sample_time),
                       states_rot_interp.interp(sample_time),
                       states_force_interp.interp(sample_time)))
        actions_interp = \
            np.hstack((actions_pos_interp.interp(sample_time),
                       actions_rot_interp.interp(sample_time),
                       actions_force_interp.interp(sample_time)))

        paired_states_actions = \
            [(s, a) for s, a in zip(states_interp, actions_interp)]

        return paired_states_actions

    def split_train_test(self, train, seed=42):
        test_len = int(len(self) * 0.2)
        print(f"test set number: {test_len}")
        lengths = [len(self) - test_len, test_len]
        train_ds, test_ds = \
            random_split(self, lengths,
                         generator=torch.Generator().manual_seed(seed))
        if train:
            print("traing dataset")
            return train_ds
        else:
            print("test dataset")
            return test_ds

    def _normalize_pos(self, data):
        """
        only normalize 3d position
        """
        mean, std = data_stats(data[:, :3])

        normalized = list(
            map(partial(normalize, mean=mean, std=std), data[:, :3]))
        return np.concatenate((normalized, data[:, 3:]), axis=1)

    def _novelty_energy(self, force_xyz, time):
        # use start phase force as baseline
        force_xyz = self._calibrate_force(force_xyz)

        # use the force magnitude sum
        force = np.sum(abs(force_xyz), axis=1)

        # low-pass filter for forces, otherwise would be too noisy
        force_lp = gaussian_filter1d(force, sigma=8)

        energy = force_lp**2  # power 2 to suppress small values
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

    def _match_start_end(self, start, end):
        """
        Assume only one contacting phase exits and it is continuous
        match the first found start and the last found end
        """
        return [start[0]], [end[-1]]
