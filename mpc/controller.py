# flake8: noqa
import numpy as np
import robosuite as suite
from types import SimpleNamespace
from robosuite.wrappers import VisualizationWrapper
from robosuite.environments.base import register_env

from mpc.dyn_model import Dyn_Model
from mpc.mpc_rollout import MPCRollout
from mpc.policy_random import Policy_Random
from mpc.helper import get_goal
from simulation.environments.peg_hole_env import PegInHoleEnv


def mpc_controller(cfg):
    params = (lambda d: SimpleNamespace(**d))(
                dict(controller_type='rand_shooting',
                     horizon=10,
                     max_step=100,
                     num_sample_seq=100,
                     rand_policy_angle_min=-0.01,
                     rand_policy_angle_max=0.01,
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

    num_eval_rollouts = 5

    for rollout_num in range(num_eval_rollouts):
        # Note: if you want to evaluate a particular goal, call env.reset with
        # a reset_state where that reset_state dict has reset_pose, reset_vel,
        # and reset_goal
        env.reset()

        env.render()
        print("action dimension", env.action_dim)
        print("action spec", env.action_spec)
        print(env.controller.action_limits)
        # Get the initial global frame state of eef
        robot_init_pos = np.copy(env.unwrapped.sim.data.body_xpos[env.peg_body_id])
        robot_init_force = np.copy(env.unwrapped.robots[0].ee_force)
        robot_init_orn = np.copy(env.unwrapped.sim.data.body_xmat[env.peg_body_id].flatten())


        # # base frame states
        # robot_init_pos = np.copy(env.unwrapped.robots[0]._hand_pos)
        # robot_init_force = np.copy(env.unwrapped.robots[0].ee_force)
        # robot_init_orn = np.copy(env.unwrapped.robots[0]._hand_orn.flatten())

        # unnormalized starting state
        starting_state = np.hstack((robot_init_pos,
                                    robot_init_force,
                                    robot_init_orn))
        print("obs", starting_state[:3])
        print(f"\n... Performing MPC rollout #{rollout_num}")

        mpc_rollout.perform_rollout(starting_state)


def build_env(controller_name='OSC_POSE',
              env_name='PegInHoleEnv',
              robots='Panda'):
    # Get controller config
    controller_config = suite.load_controller_config(
        default_controller=controller_name)
    print("controller_config", controller_config)
    # Create argument configuration
    config = {
        "env_name": env_name,
        "robots": robots,
        "controller_configs": controller_config,
    }

    register_env(PegInHoleEnv)

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="frontview",
        # render_camera=None,
        ignore_done=False,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        hard_reset=False,
    )

    env = VisualizationWrapper(env, indicator_configs=None)
    return env
