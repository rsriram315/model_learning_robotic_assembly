import numpy as np
import random
import mlflow
import os
from types import SimpleNamespace
from amira_gym_ros.task_envs.panda_model_learning_reach import PandaModelLearningReach
from amira_gym_ros.task_envs.panda_model_learning_easy_insertion import PandaModelLearningEasyInsertion
from amira_gym_ros.task_envs.panda_model_learning_hard_insertion import PandaModelLearningHardInsertion
import rospy
from mpc.dyn_model import Dyn_Model
from mpc.mpc_rollout_panda import MPCRollout
from mpc.policy_random_panda import Policy_Random
from mpc.helper import get_goal
from scipy.spatial.transform import Rotation as R

_FLOAT_EPS = np.finfo(np.float64).eps
def mpc_controller(cfg):
    # function for logging artifacts to mlflow
    def artifact_cb(filename):
        mlflow.log_artifact(filename)
    
    # function for logging params to mlflow
    def param_cb(params):
        mlflow.log_params(params)
    
    # function for logging figures to mlflow
    def image_cb(image):
        mlflow.log_figure(image, f"{params.controller_type}_rollout.eps", )
    
    # create environment and log related params to mlflow
    env, experiment = build_env(cfg)
    params = (lambda d: SimpleNamespace(**d))(
                dict(controller_type='random_shooting',
                     horizon=1,
                     max_step=200,
                     num_sample_seq=500,
                     rand_policy_angle_min=-0.02,
                     rand_policy_angle_max=0.02,
                     rand_policy_hold_action=1,
                     task_type=experiment.name,
                     record_mpc_rollouts=cfg["record_rollouts"]["record_mpc_rollout"],
                     rollout_save_dir=cfg["record_rollouts"]["save_base_dir"],))
    # logging MPC params to mlflow
    mlflow.log_params(dict(controller_type=params.controller_type,
                     horizon=params.horizon,
                     max_step=params.max_step,
                     num_sample_seq=params.num_sample_seq,
                     rand_policy_angle_min=params.rand_policy_angle_min,
                     rand_policy_angle_max=params.rand_policy_angle_max,
                     rand_policy_hold_action=params.rand_policy_hold_action,
                     task_type=experiment.name,
                     record_mpc_rollouts=cfg["record_rollouts"]["record_mpc_rollout"]))
    
    dyn_model = Dyn_Model(cfg, param_cb=param_cb)
    param_cb(dict(model=dyn_model.model))
    print("stats", dyn_model.norm.get_stats())
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
                             params,
                             artifact_cb,
                             param_cb,
                             image_cb)

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
    rospy.init_node("panda_model_learning_rollout")
    ml_logdir = os.path.join(os.environ['HOME'], '.mlflow', 'mlruns')
    mlflow.set_tracking_uri('file:' + ml_logdir)
    seed = 237
    random.seed(seed)
    np.random.seed(seed)
    if cfg["task_type"]["hard_insertion"]:
        mlflow.set_experiment('hard_insertion_experiments')
        experiment = mlflow.get_experiment_by_name("hard_insertion_experiments")
        mlflow.log_params(cfg["hard_insertion_environment"])
        env = PandaModelLearningHardInsertion(initial_position=cfg["hard_insertion_environment"]["initial_position"],
                                    target_position=cfg["hard_insertion_environment"]["target_position"],
                                    max_position_offset=cfg["hard_insertion_environment"]["max_position_offset"],
                                    nullspace_q_ref = cfg["hard_insertion_environment"]["nullspace_q_ref"],
                                    initial_quaternion = cfg["hard_insertion_environment"]["initial_quaternion"],
                                    target_quaternion = cfg["hard_insertion_environment"]["target_quaternion"],)
    
    elif cfg["task_type"]["easy_insertion"]:
        mlflow.set_experiment('easy_insertion_experiments')
        experiment = mlflow.get_experiment_by_name("easy_insertion_experiments")
        mlflow.log_params(cfg["easy_insertion_environment"])
        env = PandaModelLearningEasyInsertion(initial_position=cfg["easy_insertion_environment"]["initial_position"],
                                    target_position=cfg["easy_insertion_environment"]["target_position"],
                                    max_position_offset=cfg["easy_insertion_environment"]["max_position_offset"],
                                    nullspace_q_ref = cfg["easy_insertion_environment"]["nullspace_q_ref"],
                                    initial_quaternion = cfg["easy_insertion_environment"]["initial_quaternion"],
                                    target_quaternion = cfg["easy_insertion_environment"]["target_quaternion"],)
    elif cfg["task_type"]["reach"]:
        mlflow.set_experiment('reach_experiments')
        experiment = mlflow.get_experiment_by_name("reach_experiments")
        mlflow.log_params(cfg["reach_task_environment"])
        env = PandaModelLearningReach(initial_position=cfg["reach_task_environment"]["initial_position"],
                                    target_position=cfg["reach_task_environment"]["target_position"],
                                    max_position_offset=cfg["reach_task_environment"]["max_position_offset"],
                                    nullspace_q_ref = cfg["reach_task_environment"]["nullspace_q_ref"],
                                    initial_quaternion = cfg["reach_task_environment"]["initial_quaternion"],
                                    target_quaternion = cfg["reach_task_environment"]["target_quaternion"],)
    env.seed(seed)
    return env, experiment
