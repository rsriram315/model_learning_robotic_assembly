import numpy as np
import random
from types import SimpleNamespace
from amira_gym_ros.task_envs.panda_model_learning import PandaModelLearning
import rospy
from mpc.dyn_model import Dyn_Model
from mpc.mpc_rollout_panda import MPCRollout
from mpc.policy_random_panda import Policy_Random
from mpc.helper import get_goal
from scipy.spatial.transform import Rotation as R

_FLOAT_EPS = np.finfo(np.float64).eps
def mpc_controller(cfg):
    params = (lambda d: SimpleNamespace(**d))(
                dict(controller_type='rand_shooting',
                     horizon=1,
                     max_step=100,
                     num_sample_seq=3000,
                     rand_policy_angle_min=-0.02,
                     rand_policy_angle_max=0.02,
                     rand_policy_hold_action=1))

    dyn_model = Dyn_Model(cfg)
    print("stats", dyn_model.norm.get_stats())
    env = build_env(cfg)
    rand_policy = Policy_Random(env)
    goal_pos, goal_orn = get_goal(cfg["dataset"]["root"])
    print("goal_state ", goal_pos)
    print("goal orn mat", goal_orn)
    print("goal orn quat", (R.from_matrix(goal_orn)).as_quat())

    goal_state = goal_pos, goal_orn

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
        # Note: In Robosuite simulation if you want to evaluate a particular goal, call env.reset with
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

        print("starting state", starting_state)
        print(f"\n... Performing MPC rollout #{rollout_num}")

        mpc_rollout.perform_rollout(starting_state)


def build_env(cfg):
    # create gazebo simulation environment
    rospy.init_node("panda_model_learning_reach")
    seed = 237
    random.seed(seed)
    np.random.seed(seed)
    # specify the env name
    # env = PandaReachModelLearning(initial_position=[0.270, -0.410, 0.211], # easy insertion 0.400, 0.376, 0.400 # hard insertion 0.276, -0.414, 0.210 # 0.395, 0.373, 0.35 # reach [0.307, -0.000, 0.45]
    #                               target_position=[0.269, -0.412,  0.1825], # 0.2695, -0.4121,  0.1825 # easy insertion 0.400, 0.376,  0.285 # hard insertion 0.265, -0.41156949,  0.183  # reach [0.386, -0.008,  0.125]
    #                               max_position_offset=np.inf,
    #                               nullspace_q_ref = [-0.5593, -0.2211, -0.3533, -1.9742, -0.1533, 1.7523, 0.5162], # easy insertion 0.786, -0.058, -0.01, -1.69, -0.010, 1.64, 1.117 # hard insertion -0.5593, -0.2211, -0.3533, -1.9742, -0.1533, 1.7523, 0.5162
    #                               initial_quaternion = [0.986, -0.161, -0.0153, 0.0115], # hard insertion 0.973, -0.226, -0.041, 0.007 # easy insertion 1, 0.25, 0.000, 0
    #                               target_quaternion = [0.9849642,  -0.16776719, -0.04071259,  0.00649433], # hard insertion 0.972, -0.227, -0.045, -0.011 # easy insertion 1, 0.25, 0.000, 0
    #                               )
                                #   pause_for_train=True,) [0.3980723  0.38012593 0.31273318]
    if cfg["task_type"]["hard_insertion"]:
        env = PandaModelLearning(initial_position=cfg["hard_insertion_environment"]["initial_position"],
                                    target_position=cfg["hard_insertion_environment"]["target_position"],
                                    max_position_offset=cfg["hard_insertion_environment"]["max_position_offset"],
                                    nullspace_q_ref = cfg["hard_insertion_environment"]["nullspace_q_ref"],
                                    initial_quaternion = cfg["hard_insertion_environment"]["initial_quaternion"],
                                    target_quaternion = cfg["hard_insertion_environment"]["target_quaternion"],)
    
    elif cfg["task_type"]["easy_insertion"]:
        env = PandaModelLearning(initial_position=cfg["easy_insertion_environment"]["initial_position"],
                                    target_position=cfg["easy_insertion_environment"]["target_position"],
                                    max_position_offset=cfg["easy_insertion_environment"]["max_position_offset"],
                                    nullspace_q_ref = cfg["easy_insertion_environment"]["nullspace_q_ref"],
                                    initial_quaternion = cfg["easy_insertion_environment"]["initial_quaternion"],
                                    target_quaternion = cfg["easy_insertion_environment"]["target_quaternion"],)
    elif cfg["task_type"]["reach"]:
        env = PandaModelLearning(initial_position=cfg["reach_task_environment"]["initial_position"],
                                    target_position=cfg["reach_task_environment"]["target_position"],
                                    max_position_offset=cfg["reach_task_environment"]["max_position_offset"],
                                    nullspace_q_ref = cfg["reach_task_environment"]["nullspace_q_ref"],
                                    initial_quaternion = cfg["reach_task_environment"]["initial_quaternion"],
                                    target_quaternion = cfg["reach_task_environment"]["target_quaternion"],)
    env.seed(seed)
    return env
