import numpy as np
import h5py
from functools import partial
from pathlib import Path
from torch.utils.data import Dataset
from .data_processor import data_stats, normalize,\
                            SegmentContact, Interpolation


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
        self.states_actions = []
        self.targets = []

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
            self.states_actions[:, 0] = \
                self._normalize_pos(self.states_actions[:, 0].copy())
            # action
            self.states_actions[:, 1] = \
                self._normalize_pos(self.states_actions[:, 1].copy())

    def __len__(self):
        return len(self.states_actions)

    def __getitem__(self, idx, scale=10):
        state, action = self.states_actions[idx]
        sample = np.hstack((state[:3], action[:3])) * scale
        target = self.targets[idx, :3] * scale

        return np.float32(sample), np.float32(target)

    def _read_all_demos(self):
        for data_path in self.data_paths:
            states, actions = self._read_one_demo(data_path,
                                                  self.state_attrs,
                                                  self.action_attrs,
                                                  self.contact_only)

            # debug variables
            self.all_states_time = states["time"]
            self.all_states_force = states["force"]
            self.all_states_pos = states["pos"]
            self.all_actions_pos = actions["pos"]
            self.all_actions_time = actions["time"]

            states_actions, states_padding = \
                self._pair_state_action(self.sample_freq, states, actions)

            # use next state as target, the last target is the end state itself
            targets = states_actions[10:, 0]
            # targets = states_actions[1:, 0]
            targets = np.vstack((targets, states_padding))

            self.states_actions.extend(states_actions)
            self.targets.extend(targets)

        self.states_actions = np.array(self.states_actions)
        self.targets = np.array(self.targets)

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

    def _contact(self, time, data, is_state):
        """
        extract the contacting phase of the data and the corresponding time
        """
        time_new = []
        data_new = []
        contact_start = []
        contact_end = []

        seg_contact = SegmentContact()

        if is_state:
            force = data[:, 7:10]  #
            contact_start, contact_end = \
                seg_contact.contact_time(force, time)

            self.state_start_times = [time[s] for s in contact_start]
            self.state_end_times = [time[e] for e in contact_end]
        else:  # is_action
            contact_start = [np.argmin(abs(time - s))
                             for s in self.state_start_times]
            contact_end = [np.argmin(abs(time - e))
                           for e in self.state_end_times]

            self.action_start_times = [time[s] for s in contact_start]
            self.action_end_times = [time[e] for e in contact_end]

        for s, e in zip(contact_start, contact_end):
            time_new.append(time[s:e])
            data_new.append(data[s:e, :])

        return np.hstack(time_new), np.vstack(data_new)

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

        states_actions = \
            [(s, a) for s, a in zip(states_interp, actions_interp)]

        # padding for sliding windows
        factor = 10
        t_interval = 1.0 / self.sample_freq
        padding_time = \
            [sample_time[-1] + f * t_interval for f in range(1, factor+1)]
        states_padding = \
            np.hstack((states_pos_interp.interp(padding_time),
                       states_rot_interp.interp(padding_time),
                       states_force_interp.interp(padding_time)))

        return np.array(states_actions), np.array(states_padding)

    def _normalize_pos(self, data):
        """
        only normalize 3d position
        """
        # mean, std = data_stats(data[:, :3])
        mean, std = data_stats(data)

        # normalized = list(
        #     map(partial(normalize, mean=mean, std=std), data[:, :3]))
        normalized = list(
            map(partial(normalize, mean=mean, std=std), data))
        # return np.concatenate((normalized, data[:, 3:]), axis=1)
        return normalized
