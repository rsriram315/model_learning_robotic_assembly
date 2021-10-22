import numpy as np
import os
import random
from types import SimpleNamespace
from amira_gym_ros.task_envs.panda_model_learning_rework import PandaReachModelLearning
import rospy
import time
from mpc.dyn_model import Dyn_Model
from mpc.mpc_rollout_panda import MPCRollout
from mpc.policy_random_panda import Policy_Random
from mpc.helper import get_goal
from scipy.spatial.transform import Rotation as R


def mpc_controller(cfg):
    params = (lambda d: SimpleNamespace(**d))(
                dict(controller_type='rand_shooting',
                     horizon=10,
                     max_step=100,
                     num_sample_seq=100,
                     rand_policy_angle_min=-0.001,
                     rand_policy_angle_max=0.001,
                     rand_policy_hold_action=1))

    dyn_model = Dyn_Model(cfg)
    env = build_env()
    rand_policy = Policy_Random(env)

    goal_state = get_goal()
    mpc_rollout = MPCRollout(env,
                             dyn_model,
                             rand_policy,
                             goal_state,
                             params)

    ################################
    # RUN ROLLOUTS
    ################################

    num_eval_rollouts = 1

    for rollout_num in range(num_eval_rollouts):
        # Note: if you want to evaluate a particular goal, call env.reset with
        # a reset_state where that reset_state dict has reset_pose, reset_vel,
        # and reset_goal
        obs = env.reset()

        # Get the initial global frame state of eef
        robot_init_pos = np.copy(obs[:3])
        robot_init_force = np.copy(obs[3:6])
        robot_init_orn = np.copy(obs[6:15])
        
        # unnormalized starting state
        starting_state = np.hstack((robot_init_pos,
                                    robot_init_force,
                                    robot_init_orn))

        print(f"\n... Performing MPC rollout #{rollout_num}")

        mpc_rollout.perform_rollout(starting_state)


def build_env():
    # create gazebo simulation environment
    rospy.init_node("panda_model_learning_reach")
    seed = 237
    random.seed(seed)
    np.random.seed(seed)
    # specify the env name
    env = PandaReachModelLearning(initial_position=[0.307, 0., 0.45],
                                  target_position=[0.307, 0, 0.30],  # 0.31517366 -0.00855526  0.36535055
                                  max_position_offset=np.inf,
                                  step_wait_time=0.05)
    env.seed(seed)
    return env
