# flake8: noqa
import numpy as np
import xml.etree.ElementTree as ET
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string, find_elements, new_site
from simulation.models.objects.hole import RoundHole
from simulation.models.objects.peg import PegObj

# INIT_QPOS = np.array([-0.005, 0.487, 0.005, -2.180, -0.008, 2.676, 0.830])  # high pos
# INIT_QPOS = np.array([-0.012, 0.716, 0.010, -2.098, 0.025, 2.816, 0.857])  # low pos
# INIT_QPOS = np.array([-0.011, 0.715, 0.009, -2.095, -0.004, 2.808, 0.791])
# INIT_QPOS = np.array([0.106, 0.731, -0.058, -1.999, -0.359, 2.593, 1.188])  # tilt pos
# INIT_QPOS = np.array([-0.100, 0.753, 0.038, -1.956, 0.422, 2.524, 0.427])  # tilt pos

# INIT_QPOS = np.array([-0.011, 0.670, 0.011, -2.120, 0.002, 2.796, 0.762])  # insert pos
INIT_QPOS = np.array([-0.012, 0.248, 0.013, -2.187, 0.002, 2.433, 0.791])  # reaching task
HOLE_OFFSET = [0, 0, 0.079]
FORCE_SCALING_FACTOR = 0.00001

class PegInHoleEnv(SingleArmEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        # initialization_noise="default",
        initialization_noise=None,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        self.hole = None
        self.peg = None

        self.init_d = None
        self.init_t = None
        self.init_dist = None

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """

        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        self.hole = RoundHole(name="round-hole")
        mujoco_arena.merge(self.hole, merge_body="table")

        self.peg = PegObj(name="peg")
        # arm_name = "gripper0_eef"  # weld peg to eef
        arm_name = self.robots[0].robot_model.eef_name
        self.robots[0].robot_model.merge(self.peg, merge_body=arm_name)

        # weld peg to the arm
        elem = ET.Element('weld')
        # elem.set('body1', 'gripper0_eef')
        elem.set('body1', arm_name)
        elem.set('body2', 'peg_object')
        elem.set('solref', '0.02 1')
        self.robots[0].robot_model.equality.append(elem)

        # generate the final xml file
        # self.robots[0].robot_model.save_model('final.xml')

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.hole_body_id = self.sim.model.body_name2id(self.hole.root_body.split('_')[0] + '_object')
        self.peg_body_id = self.sim.model.body_name2id(self.peg.root_body.split('_')[0] + '_object')

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        # Run superclass reset functionality
        super()._reset_internal()
        self.robots[0].init_qpos = INIT_QPOS
        self.init_t, self.init_d, _, self.init_dist, _ = self._compute_curr_goal_diff()

    def reward(self,
               action=None,
               curr_state=None,
               goal_state=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:
            - a discrete reward of 5.0 is provided if the peg is inside the hole
              - Note that we enforce that it's inside at an appropriate angle (cos(theta) > 0.95).

        Un-normalized summed components if using reward shaping:
            - Reaching: in [0, 1], to encourage the arms to approach each other
            - Perpendicular Distance: in [0,1], to encourage the arms to approach each other
            - Parallel Distance: in [0,1], to encourage the arms to approach each other
            - Alignment: in [0, 1], to encourage having the right orientation between the peg and hole.

        Note that the final reward is normalized and scaled by reward_scale / 5.0 as
        well so that the max score is equal to reward_scale
        """

        reward = 0

        if curr_state is None and goal_state is None:
            t, d, cos, dist, force_mag = self._compute_curr_goal_diff()
        else:
            curr_pos = curr_state[:3]
            curr_force = curr_state[3:6]
            curr_orn = curr_state[6:15].reshape((3, 3))

            goal_pos, _ = goal_state
            goal_orn = np.eye(3)

            # Grab relevant values
            t, d, cos = self._compute_orientation((curr_pos, curr_orn),
                                                  (goal_pos, goal_orn))
            # reaching reward, with coord w.r.t. table as origin
            # added offset to make successful insertion reward = 1
            dist = np.linalg.norm(curr_pos - goal_pos)
            force_mag = FORCE_SCALING_FACTOR * np.linalg.norm(curr_force)

        reaching_reward = 1 - np.tanh(dist / self.init_dist)
        reward += reaching_reward

        force_reward = 1 - force_mag
        reward += force_reward

        # Orientation reward
        reward += 1 - np.tanh(d / self.init_d)
        reward += 1 - np.tanh(t / self.init_t)
        reward += cos

        if self.reward_scale is not None:
            reward *= self.reward_scale / 5.0
        return reward

    def _post_action(self, action):
        reward, self.done, _ = super()._post_action(action)

        # Check if already reach desired location
        if not self.ignore_done and self._check_success():
            self.done = True

        return reward, self.done, {}

    def _check_success(self):
        """
        Check if peg is successfully aligned and placed within the hole

        Returns:
            bool: True if peg is placed in hole correctly
        """
        d, t, cos, _, _ = self._compute_curr_goal_diff()

        return d < 0.0004 and t <= 0.0003 and cos > 0.999

    def _compute_curr_goal_diff(self):
        curr_pos = self.sim.data.body_xpos[self.peg_body_id]
        curr_force = self.robots[0].ee_force
        curr_orn = self.sim.data.body_xmat[self.peg_body_id]
        curr_orn.shape = (3, 3)

        goal_pos = self.sim.data.body_xpos[self.hole_body_id] + HOLE_OFFSET
        goal_orn = self.sim.data.body_xmat[self.hole_body_id]
        goal_orn.shape = (3, 3)

        d, t, cos = self._compute_orientation((curr_pos, curr_orn),
                                              (goal_pos, goal_orn))

        dist = np.linalg.norm(curr_pos - goal_pos)
        force_mag = FORCE_SCALING_FACTOR * np.linalg.norm(curr_force)
        return d, t, cos, dist, force_mag

    def _compute_orientation(self, curr_state, goal_state):
        """
        Helper function to return the relative positions between the hole and the peg.
        In particular, the intersection of the line defined by the peg and the plane
        defined by the hole is computed; the parallel distance, perpendicular distance,
        and angle are returned.

        Returns:
            3-tuple:
                - (float): parallel distance
                - (float): perpendicular distance
                - (float): angle
        """

        # The quantities in mjData that start with "x" are expressed in global coordinates
        # Frame orientations are usually stored as 3-by-3 matrices (xmat)

        # What we doing here is, to try to calculate the parallel and perpendicular distance
        # between plane of the hole's center and the peg's center (by factorizing the euclidean distance).
        curr_pos, curr_orn = curr_state
        goal_pos, goal_orn = goal_state

        # first transform the plane's normal [0, 0, 1],
        # such that hole's plane is parallel to peg
        # just multiply by the rotation matrix of peg
        v = curr_orn @ np.array([0, 0, 1])
        v = v / np.linalg.norm(v)  # v is the normal of the hole plane

        # t (parallel distance) is just the cosine * euclidean distance
        t = (goal_pos - curr_pos) @ v / (np.linalg.norm(v) ** 2)
        # d (vertical distance) is just the sine * euclidean distance
        d = np.linalg.norm(np.cross(v, curr_pos - goal_pos)) / np.linalg.norm(v)

        goal_normal = goal_orn @ np.array([0, 0, 1])
        return (
            abs(t),
            abs(d),
            abs(np.dot(goal_normal, v) / np.linalg.norm(goal_normal) / np.linalg.norm(v)),
            # this is reward, the larger the dot prod of normals, the closer
        )
