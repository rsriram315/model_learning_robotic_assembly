from amira_gym_ros import robot_interfaces
import numpy as np
from gym import spaces
from numpy.core.numeric import empty_like
import rospy
import scipy.spatial.transform as trf
from scipy.spatial.transform import Rotation as R
from amira_gym_ros.task_envs import RobotEnv
from amira_arm_control_msgs.msg import CartesianImpedanceSetpoint
from geometry_msgs.msg import WrenchStamped
from amira_utils.transformation_utils import rotation_distance, Transform
from amira_gym_ros.robot_interfaces import get_robot_interface
from random_shooting_panda import RandomShooting

rospy.init_node("test")

# create publisher on the impedance controller topic with setpoint messages
_cis_publisher = rospy.Publisher('/PandaCartesianImpedanceController/controllerReference',
                                        CartesianImpedanceSetpoint)
robot_interface = get_robot_interface('panda')
robot_interface.cm_helpers.activate_controllers(
    ['PandaStatePublisher', 'PandaCartesianImpedanceController'])


for i in range(5):

        # create the message
        msg = CartesianImpedanceSetpoint()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "panda_link0"

        msg.pose.position.x = robot_interface._arm_state.tcp_pose_base.position.x
        msg.pose.position.y = robot_interface._arm_state.tcp_pose_base.position.y
        msg.pose.position.z = 0.4 - i * 0.01
        msg.pose.orientation.x = robot_interface._arm_state.tcp_pose_base.orientation.x
        msg.pose.orientation.y = robot_interface._arm_state.tcp_pose_base.orientation.y
        msg.pose.orientation.z = robot_interface._arm_state.tcp_pose_base.orientation.z
        msg.pose.orientation.w = robot_interface._arm_state.tcp_pose_base.orientation.w
        msg.pose_ref_frame = "panda_link0"

        print('msg:')
        print(msg)

        # publish the message
        _cis_publisher.publish(msg)
        rospy.sleep(1)

