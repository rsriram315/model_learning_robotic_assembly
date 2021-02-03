import numpy as np
import h5py
import torch
from functools import partial
from pathlib import Path
from torch.utils.data import Dataset, random_split
from .data_processor import data_stats, normalize, quaternion_to_axis_angle,\
                            axis_angle_to_euler_vector, Interpolation


class DemoDataset(Dataset):
    def __init__(self, root, fnames, sample_freq,
                 state_attrs, action_attrs,
                 preprocess):
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

        self.sample_freq = sample_freq
        self.state_attrs = state_attrs
        self.action_attrs = action_attrs

        self.all_states = []
        self.all_actions = []
        self.all_states_time = []
        self.all_actions_time = []
        self.sample_time = []
        self.paired_states_actions = []

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

    def _read_all_demos(self):
        for data_path in self.data_paths:
            states_time, actions_time = self._read_one_demo(data_path,
                                                            self.state_attrs,
                                                            self.action_attrs)
            if self.preprocess["euler_vector"]:
                states_time["data"] = self._quaternion_to_euler_vector(
                                        np.asarray(states_time["data"]))
                actions_time["data"] = self._quaternion_to_euler_vector(
                                            np.asarray(actions_time["data"]))

            self.all_states.extend(states_time["data"][:])
            self.all_actions.extend(actions_time["data"][:])
            self.all_states_time.extend(states_time["time"][:])
            self.all_actions_time.extend(actions_time["time"][:])

            self._pair_state_action(self.sample_freq,
                                    states_time,
                                    actions_time)

        self.all_states = np.asarray(self.all_states)
        self.all_actions = np.asarray(self.all_actions)
        self.paired_states_actions = np.asarray(self.paired_states_actions)

    def _read_one_demo(self, data_path, states, actions, time="time_stamp"):
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

            data_time = [{}, {}]  # [[state, time], [action, time]]
            for n, i in enumerate([states, actions]):
                # clip time to avoid negative value
                t = np.clip(np.asarray(f[i["name"]][time]), 0, None)

                start_idx = np.argmin(abs(t - grasp_start_t))
                stop_idx = np.argmin(abs(t - grasp_stop_t))
                # discard the data when the arm not grasping the object
                d = np.array(f[i["name"]][i["attrs"]])[start_idx:stop_idx]
                t = t[start_idx:stop_idx]

                data_time[n]["data"] = d
                data_time[n]["time"] = t
        return data_time

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

    def _pair_state_action(self, sample_freq, states_time, actions_time):
        """
        form state action pair, state has more dense data than action.
        choosing the action which is closest to the state.
        """
        start_time = max(min(states_time["time"]), min(actions_time["time"]))
        end_time = min(max(states_time["time"]), max(actions_time["time"]))
        print(f"dataset start time {start_time}, end time {end_time}")

        sample_time = np.arange(start_time, end_time, 1.0/sample_freq)
        self.sample_time.extend(sample_time)
        print(f"there are {sample_time.shape[0]} samples")

        # interpolation, position only
        state_interp = Interpolation(states_time,
                                     self.preprocess["interpolation"])
        action_interp = Interpolation(actions_time,
                                      self.preprocess["interpolation"])
        for t in sample_time:
            state = state_interp.interp(t)
            action = action_interp.interp(t)
            self.paired_states_actions.append([state, action])

    def split_train_test(self, train, seed=42):
        test_len = int(len(self)*0.2)
        print(f"test set number: {test_len}")
        lengths = [len(self) - test_len, test_len]
        train_ds, test_ds = \
            random_split(self, lengths,
                         generator=torch.Generator().manual_seed(seed))
        if train:
            print("traing dataset")
            return train_ds
        else:
            return test_ds

    def _normalize_pos(self, data):
        """
        only normalize 3d position
        """
        mean, std = data_stats(data[:, :3])

        normalized = list(map(partial(normalize, mean=mean, std=std),
                              data[:, :3]))
        return np.concatenate((normalized, data[:, 3:]), axis=1)

    def _quaternion_to_euler_vector(self, data):
        """
        transform quaternion x, y, z, w to euler vector
        """
        axis_angle = list(map(quaternion_to_axis_angle, data[:, 3:]))
        euler_vec = list(map(axis_angle_to_euler_vector, axis_angle))

        return np.concatenate((data[:, :3], euler_vec), axis=1)
