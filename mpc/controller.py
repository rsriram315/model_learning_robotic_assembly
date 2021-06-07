import robosuite as suite
import numpy as np
from types import SimpleNamespace
from robosuite.wrappers import VisualizationWrapper

from simulation.environments.peg_hole_env import PegInHoleEnv  # noqa
from mpc.dyn_model import Dyn_Model
from mpc.mpc_rollout import MPCRollout
from mpc.policy_random import Policy_Random


def mpc_controller(cfg):
    params = (lambda d: SimpleNamespace(**d))(
        dict(K=1,
             rollout_length=200,
             horizon=20,
             controller_type='rand',
             num_sample_seq=200,
             sample_rot=False,
             rand_policy_angle_min=0,
             rand_policy_angle_max=0,
             rand_policy_hold_action=1))

    dyn_model = Dyn_Model(cfg)

    env = _build_env()
    rand_policy = Policy_Random(env.unwrapped)

    mpc_rollout = MPCRollout(env,
                             dyn_model,
                             rand_policy,
                             params)

    ################################
    # RUN ROLLOUTS
    ################################

    list_rewards = []
    # list_scores = []
    rollouts = []
    num_eval_rollouts = 1

    for rollout_num in range(num_eval_rollouts):
        # Note: if you want to evaluate a particular goal, call env.reset with
        # a reset_state where that reset_state dict has reset_pose, reset_vel,
        # and reset_goal
        starting_observation = env.reset()

        env.render()

        starting_state = np.hstack(
            (env.unwrapped.robots[0]._hand_pos,
             env.unwrapped.robots[0].ee_force,
             env.unwrapped.robots[0]._hand_orn.flatten()))

        print(f"\n... Performing MPC rollout #{rollout_num}")

        rollout_info = mpc_rollout.perform_rollout(
            starting_state,
            starting_observation,
            controller_type=params.controller_type)

        # save info from MPC rollout
        list_rewards.append(rollout_info['rollout_rewardTotal'])
        # list_scores.append(rollout_info['rollout_meanFinalScore'])
        rollouts.append(rollout_info)


def _build_env(controller_name='OSC_POSE',
               env_name='PegInHoleEnv',
               robots='Panda'):
    # Get controller config
    controller_config = suite.load_controller_config(
        default_controller=controller_name)

    # Create argument configuration
    config = {
        "env_name": env_name,
        "robots": robots,
        "controller_configs": controller_config,
    }

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=False,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        hard_reset=False,
    )
    env = VisualizationWrapper(env, indicator_configs=None)
    return env
