"""
This file implements a wrapper for saving simulation states to disk.
This data collection wrapper is useful for collecting demonstrations.
"""
# flake8: noqa
import os
import h5py
import numpy as np
import robosuite.utils.transform_utils as T
from datetime import datetime

STATE = "PandaStatePublisherarm_states"
ACTION = "PandaCartesianImpedanceControllercontrollerReference"
GRIPPER = "franka_gripperjoint_states"


class DataCollection:
    def __init__(self, env, directory, collect_freq=1, flush_freq=100):
        """
        Initializes the data collection wrapper.

        Args:
            env (MujocoEnv): The environment to monitor.
            directory (str): Where to store collected data.
            collect_freq (int): How often to save simulation state, in terms of environment steps.
            flush_freq (int): How frequently to dump data to disk, in terms of environment steps.
        """

        # the base directory for all logging
        self.env = env
        self.directory = directory
        self.recording = None
        self.num_recording = 0

        self.base_pos = self.env.unwrapped.robots[0].base_pos
        self.base_ori = self.env.unwrapped.robots[0].base_ori  # quaternions

        # how often to save simulation state, in terms of environment steps
        self.collect_freq = collect_freq

        # how frequently to dump data to disk, in terms of environment steps
        self.flush_freq = flush_freq

        if not os.path.exists(directory):
            print("DataCollectionWrapper: making new directory at {}".format(directory))
            os.makedirs(directory)

    def flush(self):
        """
        Method to flush internal state to disk.
        """
        now = datetime.now()
        filename = f"recording_{self.num_recording:04}_{now.year}_{now.month:02}_{now.day:02}_{now.hour:02}{now.minute:02}.h5"
        file_path = os.path.join(self.directory, filename)

        for key in self.recording.keys():
            with h5py.File(file_path, 'a') as f:
                # check whether a group with the resource name already exists,
                # if not create it and set up its children datasets
                group = f.create_group(key)
                data = self.recording[key]

                for key_1 in data.keys():
                    data_1 = self.recording[key][key_1]
                    group.create_dataset(key_1, data=data_1)

        self.num_recording += 1
        self.reset()

    def reset(self):
        """
        Extends vanilla reset() function call to accommodate data collection

        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        self.recording = {"PandaStatePublisherarm_states": {
                            "time_stamp": [],
                            "tcp_pose_base": [],
                            "tcp_wrench_ee": []},

                          "PandaCartesianImpedanceControllercontrollerReference": {
                            "time_stamp": [],
                            "pose": [],
                            "wrench": []},

                          "franka_gripperjoint_states": {
                            "time_stamp": [],
                            "q": []}}

    def record(self, curr_pose, action_pose, curr_wrench, curr_time):
        # collect the current simulation state if necessary
        # if self.t % self.collect_freq == 0:

        curr_pos = curr_pose[:3, 3]
        curr_orn = T.mat2quat(np.copy(curr_pose[:3, :3]))

        action_pos = action_pose[:3, 3]
        action_orn = T.mat2quat(np.copy(action_pose[:3, :3]))

        # gripper state
        self.recording[GRIPPER]["time_stamp"].append(curr_time)
        self.recording[GRIPPER]["q"].append(curr_time)

        # transform to base frame is still needed
        self.recording[STATE]["time_stamp"].append(curr_time)
        # https://robosuite.ai/docs/simulation/robot.html heres said that the _hand_pos and _hand_quat
        # are the position and orientation of the end-effector in base frame of the robot
        self.recording[STATE]["tcp_pose_base"].append(np.hstack((curr_pos, curr_orn)))
        # np.hstack((self.env.unwrapped.robots[0]._hand_pos, self.env.unwrapped.robots[0]._hand_quat)))
        self.recording[STATE]["tcp_wrench_ee"].append(curr_wrench)

        # transform to base frame and also express the orientation in changes
        self.recording[ACTION]["time_stamp"].append(curr_time)
        self.recording[ACTION]["pose"].append(np.hstack((action_pos, action_orn)))
        # orientation still needed to be convert to quaternions
        self.recording[ACTION]["wrench"].append([0, 0, 0, 0, 0, 0])
