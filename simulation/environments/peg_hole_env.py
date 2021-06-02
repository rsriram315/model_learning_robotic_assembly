# flake8: noqa
import numpy as np
import xml.etree.ElementTree as ET
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask

from simulation.models.objects.hole import RoundHole
from simulation.models.objects.peg import PegObj

INIT_QPOS = np.array([-0.011, 0.670, 0.011, -2.120, 0.002, 2.796, 0.762])
HOLE_OFFSET = [0, 0, 0.79]

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
        arm_name = "gripper0_eef"  # weld peg to eef
        self.robots[0].robot_model.merge(self.peg, merge_body=arm_name)

        # weld peg to the arm
        elem = ET.Element('weld')
        elem.set('body1', 'gripper0_eef')
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

    def reward(self, action=None):
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

        Note that the final reward is normalized and scaled by reward_scale / 4.0 as
        well so that the max score is equal to reward_scale
        """

        reward = 0

        # Right location and angle
        if self._check_success():
            reward = 1.0

        # use a shaping reward
        if self.reward_shaping:
            # Grab relevant values
            t, d, cos = self._compute_orientation()
            # reaching reward, with coord w.r.t. table as origin
            # added offset to make successful insertion reward = 1
            hole_pos = self.sim.data.body_xpos[self.hole_body_id] + HOLE_OFFSET
            peg_pos = self.sim.data.body_xpos[self.peg_body_id]
            dist = np.linalg.norm(peg_pos - hole_pos)
            reaching_reward = 1 - np.tanh(1.0 * dist)
            reward += reaching_reward

            # Orientation reward
            reward += 1 - np.tanh(d)
            reward += 1 - np.tanh(np.abs(t))
            reward += cos

        # if we're not reward shaping, scale sparse reward so that the max reward is identical to its dense version
        else:
            reward *= 4.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 4.0

        return reward

    def _check_success(self):
        """
        Check if peg is successfully aligned and placed within the hole

        Returns:
            bool: True if peg is placed in hole correctly
        """
        t, d, cos = self._compute_orientation()

        return d < 0.08 and -0.00055 <= t <= 0.00055 and cos > 0.999

    def _compute_orientation(self):
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

        peg_mat = self.sim.data.body_xmat[self.peg_body_id]
        peg_pos = self.sim.data.body_xpos[self.peg_body_id]
        peg_mat.shape = (3, 3)

        hole_pos = self.sim.data.body_xpos[self.hole_body_id]
        hole_mat = self.sim.data.body_xmat[self.hole_body_id]
        hole_mat.shape = (3, 3)

        # first transform the plane's normal [0, 0, 1],
        # such that hole's plane is parallel to peg
        # just multiply by the rotation matrix of peg
        v = peg_mat @ np.array([0, 0, 1])
        v = v / np.linalg.norm(v)  # v is the normal of the hole plane
        center = hole_pos

        # t (parallel distance) is just the cosine * euclidean distance
        t = (center - peg_pos) @ v / (np.linalg.norm(v) ** 2)
        # d (vertical distance) is just the sine * euclidean distance
        d = np.linalg.norm(np.cross(v, peg_pos - center)) / np.linalg.norm(v)

        hole_normal = hole_mat @ np.array([0, 0, 1])
        return (
            t,
            d,
            abs(np.dot(hole_normal, v) / np.linalg.norm(hole_normal) / np.linalg.norm(v)),
            # this is reward, the larger the dot prod of normals, the closer
        )
