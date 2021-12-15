import numpy as np
import random
from types import SimpleNamespace
from amira_gym_ros.task_envs.panda_model_learning_rework import PandaReachModelLearning
import rospy
from mpc.dyn_model import Dyn_Model
from mpc.mpc_rollout_panda import MPCRollout
from mpc.policy_random_panda import Policy_Random
from mpc.helper import get_goal

_FLOAT_EPS = np.finfo(np.float64).eps
def mpc_controller(cfg):
    params = (lambda d: SimpleNamespace(**d))(
                dict(controller_type='rand_shooting',
                     horizon=10,
                     max_step=100,
                     num_sample_seq=3000,
                     rand_policy_angle_min=-0.01,
                     rand_policy_angle_max=0.01,
                     rand_policy_hold_action=1))

    dyn_model = Dyn_Model(cfg)
    print("stats", dyn_model.norm.get_stats())
    env = build_env()
    rand_policy = Policy_Random(env)

    goal_pos, goal_orn = get_goal()
    print("goal_state ", goal_pos)
    print("goal orn", goal_orn)
    goal_pos_norm = (goal_pos - np.array([0.29454313, -0.02973482,  0.02580059])) / np.array([1.07804996e-01, 4.36442115e-02, 4.49540457e-01])
    goal_pos_norm = 2 * (goal_pos_norm - 0.5)
    
    print("goal_state norm", goal_pos_norm) #[ 0.69785281 -0.01056377 -0.56079903]

    goal_pos_inv_norm  = (goal_pos / 2 + 0.5) * (np.array([1.07804996e-01, 4.36442115e-02, 4.49540457e-01]) + _FLOAT_EPS)
    goal_pos_inv_norm = goal_pos_inv_norm + np.array([0.29454313, -0.02973482,  0.02580059])
    
    print("goal_state inv_norm", goal_pos_inv_norm) #[ 0.36925531 -0.00809042  0.27855918]

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

        print("starting state", starting_state)
        print(f"\n... Performing MPC rollout #{rollout_num}")

        mpc_rollout.perform_rollout(starting_state)


def build_env():
    # create gazebo simulation environment
    rospy.init_node("panda_model_learning_reach")
    seed = 237
    random.seed(seed)
    np.random.seed(seed)
    # specify the env name
    env = PandaReachModelLearning(initial_position=[0.307, -0.000, 0.45],
                                  target_position=[0.386, -0.008,  0.125],  #panda reach new  0.386, -0.008,  0.124
                                  max_position_offset=np.inf,   # goal panda sideways 0.39650042, 0.35287725, 0.2123521
                                  pause_for_train=True,)
    env.seed(seed)
    return env
