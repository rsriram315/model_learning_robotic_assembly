import h5py
import numpy as np
from utils import grasp_time
from torch.utils import data
from torch.utils.data import DataSet, dataset


class DemoDataSet(DataSet):
    def __init__(self, data_path):
        super().__init__()

    def read_data(data_path):
        with h5py.File(data_path, 'r') as f:
            cam1_dpt_t = np.array(f['realsense_1aligned_depth_to_colorimage_raw']['time_stamp'])
            cam2_dpt_t = np.array(f['realsense_2aligned_depth_to_colorimage_raw']['time_stamp'])

            cam1_clr_t = np.array(f['realsense_1colorimage_raw']['time_stamp'])
            cam2_clr_t = np.array(f['realsense_2colorimage_raw']['time_stamp'])

            arm_state_t = np.array(f['PandaStatePublisherarm_states']['time_stamp'])
            gripper_t = np.array(f['franka_gripperjoint_states']['time_stamp'])
            controller_t = np.array(f['PandaCartesianImpedanceControllercontrollerReference']['time_stamp'])

            gripper_s = np.array(f['franka_gripperjoint_states']['q'])
            start_t, stop_t = grasp_time(gripper_s, gripper_t)

    
states: end-effector position without orientation
action: controller


