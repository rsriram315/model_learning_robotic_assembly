"""
This file implements a wrapper for saving simulation states to disk.
This data collection wrapper is useful for collecting demonstrations
obtained by executing MPC on the task. The collected demonstarations
can be used for fine tuning the Forward model.
"""
# flake8: noqa
import os
import h5py
import numpy as np
from datetime import datetime

STATE = "PandaStatePublisherarm_states"
ACTION = "PandaCartesianImpedanceControllercontrollerReference"
GRIPPER = "franka_gripperjoint_states"
ROLLOUT = "mpc_rollout_states"

class DataCollection:
    def __init__(self, env, directory, artifact_cb, collect_freq=1, flush_freq=100):
        """
        Initializes the data collection wrapper.

        Args:
            env (MujocoEnv): The environment to monitor.
            directory (str): Where to store collected data.
            artifact_cb : mlflow callback function to save .h5 file as mlflow artifact data
            collect_freq (int): How often to save simulation state, in terms of environment steps.
            flush_freq (int): How frequently to dump data to disk, in terms of environment steps.
        """

        # the base directory for all logging
        self.env = env
        self.directory = directory
        self.recording = None
        self.num_recording = 0
        self.artifact_cb = artifact_cb

        print("\n Intializing data collection \n")

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
        n = len(os.listdir(self.directory))
        now = datetime.now()
        self.filename = f"recording_{n+1:04}_{now.year}_{now.month:02}_{now.day:02}_{now.hour:02}_{now.minute:02}.h5"
        print(f"\n... Saving data to {self.filename} ...\n")
        self.file_path = os.path.join(self.directory, self.filename)

        for key in self.recording.keys():
            with h5py.File(self.file_path, 'a') as f:
                # check whether a group with the resource name already exists,
                # if not create it and set up its children datasets
                group = f.create_group(key)
                data = self.recording[key]

                for key_1 in data.keys():
                    data_1 = self.recording[key][key_1]
                    group.create_dataset(key_1, data=data_1)

        self.artifact_cb(self.file_path)
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
                            "q": []},
                        
                          "mpc_rollout_states":{
                              "time_stamp": [],
                              "actual_next_state": [],
                              "pred_next_state": [],
                              "reward": []}
                          }

    def record(self, curr_state_pose, curr_action_pose, curr_state_wrench, curr_time,
               actual_next_state, pred_next_state, reward):

        # gripper state
        self.recording[GRIPPER]["time_stamp"].append(curr_time)
        self.recording[GRIPPER]["q"].append(curr_time)

        # store robot state info read from /PandaStatePublisher/arm_states
        self.recording[STATE]["time_stamp"].append(curr_time)
        self.recording[STATE]["tcp_pose_base"].append(curr_state_pose)

        self.recording[STATE]["tcp_wrench_ee"].append(curr_state_wrench)

        # store the messages sent to CartesianImpedenceSetpoint
        self.recording[ACTION]["time_stamp"].append(curr_time)
        self.recording[ACTION]["pose"].append(curr_action_pose)
        self.recording[ACTION]["wrench"].append([0, 0, 0, 0, 0, 0])

         # store the mpc states for visualisation
        self.recording[ROLLOUT]["time_stamp"].append(curr_time)
        self.recording[ROLLOUT]["actual_next_state"].append(actual_next_state)
        self.recording[ROLLOUT]["pred_next_state"].append(pred_next_state)
        self.recording[ROLLOUT]["reward"].append(reward)
