import argparse
import numpy as np
from numpy.lib.function_base import select
import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite import load_controller_config
from robosuite.wrappers import VisualizationWrapper
from robosuite.environments.base import register_env

from simulation.utils.input import input2action
from simulation.utils.data_collection import DataCollection
from simulation.environments.peg_hole_env import PegInHoleEnv
# from simulation.models.robots.manipulators import bosch_robot

# flake8: noqa

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="PegInHoleEnv")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda",
                        help="Which robot(s) to use in the env")
    # parser.add_argument("--robots", nargs="+", type=str, default="Bosch",
    #                     help="Which robot(s) to use in the env")
    parser.add_argument("--config", type=str, default="single-arm-opposed",
                        help="Specified environment configuration if necessary")
    parser.add_argument("--arm", type=str, default="right",
                        help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--switch-on-grasp", action="store_true",
                        help="Switch gripper control on gripper action")
    parser.add_argument("--toggle-camera-on-grasp", action="store_true",
                        help="Switch camera angle on gripper action")
    parser.add_argument("--device", type=str, default="spacemouse")
    parser.add_argument("--pos-sensitivity", type=float, default=.3,
                        help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=.3,
                        help="How much to scale rotation user inputs")
    args = parser.parse_args()

    # Gripper moves sequentially and linearly in x, y, z direction,
    # then sequentially rotates in x-axis, y-axis, z-axis, relative
    # to the global coordinate frame (i.e., static / camera frame of reference)
    controller_name = 'OSC_POSE'

    # Get controller config
    controller_config = load_controller_config(default_controller=controller_name)

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment
    # and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config
    else:
        args.config = None

    register_env(PegInHoleEnv)

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="frontview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        hard_reset=False,
    )

    # Wrap this environment in a visualization wrapper
    data_dir = "data/data_reach"
    env = VisualizationWrapper(env, indicator_configs=None)
    data_collector = DataCollection(env, data_dir)

    # Setup printing options for numbers
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(pos_sensitivity=args.pos_sensitivity,
                          rot_sensitivity=args.rot_sensitivity)
        env.viewer.add_keypress_callback("any", device.on_press)
        env.viewer.add_keyup_callback("any", device.on_release)
        env.viewer.add_keyrepeat_callback("any", device.on_press)
    elif args.device == "spacemouse":
        from simulation.devices.spacemouse import SpaceMouse

        device = SpaceMouse(pos_sensitivity=args.pos_sensitivity,
                            rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception(
            "Invalid device choice: choose either 'keyboard' or 'spacemouse'."
        )

    while True:
        # Reset the environment
        obs = env.reset()

        # Setup rendering
        cam_id = 0
        num_cam = len(env.sim.model.camera_names)
        env.render()

        # Initialize variables that should the maintained between resets
        last_grasp = 0

        # Initialize device control
        device.start_control()
        data_collector.reset()

        # Get the initial position of eef
        robot_init_pos = env.unwrapped.robots[0]._hand_pos

        while True:
            # Set active robot
            active_robot = (env.robots[0] if args.config == "bimanual"
                            else env.robots[args.arm == "left"])

            ##############################################################
            # The spacemouse/keyboard generates delta movement in ee_frame
            ##############################################################

            # Get the newest action
            action, action_G_pose, grasp = input2action(
                device=device,
                robot=active_robot,
                robot_init_pos=robot_init_pos,
                active_arm=args.arm,
                env_configuration=args.config
            )
            # action: x, y, z, roll, pitch, yaw

            # If action is none, then this a reset so we should break
            if action is None:
                break

            # If the current grasp is active (1) and last grasp is not (-1)
            # (i.e.: grasping input just pressed),
            # toggle arm control and / or camera viewing angle if requested
            if last_grasp < 0 < grasp:
                if args.switch_on_grasp:
                    args.arm = "left" if args.arm == "right" else "right"
                if args.toggle_camera_on_grasp:
                    cam_id = (cam_id + 1) % num_cam
                    env.viewer.set_camera(camera_id=cam_id)
            # Update last grasp
            last_grasp = grasp

            # Fill out the rest of the action space if necessary
            rem_action_dim = env.action_dim - action.size
            if rem_action_dim > 0:
                # Initialize remaining action space
                rem_action = np.zeros(rem_action_dim)
                # This is a multi-arm setting, choose which arm to control
                # and fill the rest with zeros
                if args.arm == "right":
                    action = np.concatenate([action, rem_action])
                elif args.arm == "left":
                    action = np.concatenate([rem_action, action])
                else:
                    # Only right and left arms supported
                    print("Error: Unsupported arm specified -- "
                          "must be either 'right' or 'left'! Got: {}"
                          .format(args.arm))
            elif rem_action_dim < 0:
                # We're in an environment with no gripper action space,
                # so trim the action space to be the action dim
                action = action[:env.action_dim]

            ##########################################################################
            # transform the action signal from spacemouse in global frame to base frame
            ##########################################################################

            curr_G_pos = np.copy(env.unwrapped.sim.data.body_xpos[env.peg_body_id])
            curr_G_orn = np.copy(env.unwrapped.sim.data.body_xmat[env.peg_body_id]).reshape((3, 3))
            curr_G_pose = T.make_pose(curr_G_pos, curr_G_orn)

            # the position is set point in global frame, while the rotation is the delta rotation
            action_G_pose = action_G_pose.dot(curr_G_pose)

            curr_wrench = np.hstack((env.unwrapped.robots[0].ee_force,
                                     env.unwrapped.robots[0].ee_torque))

            curr_time = env.unwrapped.cur_time

            ########################################################
            # However, the env takes the action as in world frame
            # TODO: which we need to do transformation to account to that!
            ########################################################

            # Step through the simulation and render
            obs, reward, done, info = env.step(action)

            ###################
            # Debug information
            ###################
            # print(f'current pose: {curr_G_pos}')
            # print(f'action: {action[:3]}')
            # print(f'target pos: {action_pos}'
            # print(f"setpt {G_in_B[:3, -1]}")
            # print(f"state {env.unwrapped.robots[0]._hand_pos}")
            # print(f"setptori {action_ori}")
            # print(f"stateori {env.unwrapped.robots[0]._hand_quat}")
            # print(f"eef force {env.unwrapped.robots[0].ee_force}")
            # print(f"{env.unwrapped.robots[0]._joint_positions}")
            # print(f"eef force {np.linalg.norm(env.unwrapped.robots[0].ee_force)}\n")
            # print(env.unwrapped.robots[0]._joint_positions)
            # print(f'ground truth pos {env.unwrapped.sim.data.body_xpos[env.peg_body_id]}\n')
            # print(f"reward is {reward}\n")

            # Record state and action
            if not device.get_controller_state()["reset"]:
                data_collector.record(curr_G_pose, action_G_pose, curr_wrench, curr_time)
            else:
                data_collector.flush()
                break

            env.render()
