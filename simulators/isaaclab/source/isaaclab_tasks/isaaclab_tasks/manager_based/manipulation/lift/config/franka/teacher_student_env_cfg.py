# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for teacher-student lift-cube RL."""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

from isaaclab_tasks.manager_based.manipulation.lift import mdp

from . import joint_pos_env_cfg


@configclass
class ObservationsCfg:
    """Student proprioception only."""

    @configclass
    class PolicyCfg(ObsGroup):
        student_state = ObsTerm(func=mdp.student_proprio_observation)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """KL training does not use environment rewards."""

    pass


@configclass
class FrankaCubeLiftTeacherStudentEnvCfg(joint_pos_env_cfg.FrankaCubeLiftEnvCfg):
    """Lift-cube environment for proprio teacher-student RL with 7D twist actions."""

    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            scale=0.04,
            use_default_offset=True,
        )

        self.commands.object_pose.debug_vis = False
        self.scene.ee_frame.debug_vis = False
        # Keep the lift target fixed for the whole episode (default command resamples every 5 s).
        self.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)

        self.events.reset_object_position.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
        }
        # Fixed lift goal (midpoint of default Isaac-Lift-Cube command ranges).
        self.commands.object_pose.ranges.pos_x = (0.5, 0.5)
        self.commands.object_pose.ranges.pos_y = (0.0, 0.0)
        self.commands.object_pose.ranges.pos_z = (0.375, 0.375)

        self.curriculum.action_rate = None
        self.curriculum.joint_vel = None

        self.observations.policy.enable_corruption = False
        self.observations.policy.concatenate_terms = True

        # Twist->IK-Rel tracking is slower than direct oracle IK commands.
        self.episode_length_s = 30.0


@configclass
class FrankaCubeLiftTeacherStudentEnvCfg_PLAY(FrankaCubeLiftTeacherStudentEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5


@configclass
class FrankaCubeLiftOracleTwistPlayEnvCfg(FrankaCubeLiftTeacherStudentEnvCfg):
    """Teacher-student IK-Rel env with cameras for oracle twist visualization."""

    def __post_init__(self):
        super().__post_init__()
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import CameraCfg

        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5

        self.scene.ee_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/ee_camera",
            update_period=0.0,
            height=84,
            width=84,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.13, 0.0, -0.24),
                rot=(-0.70614, 0.03701, 0.03701, -0.70614),
                convention="ros",
            ),
        )
        self.scene.third_person_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/third_person_camera",
            update_period=0.0,
            height=84,
            width=84,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 5.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(2.2, 0.0, 1.05),
                rot=(0.35355, -0.61237, -0.61237, 0.35355),
                convention="ros",
            ),
        )
        self.num_rerenders_on_reset = 3
        # Twist execution via IK-Rel is slower than direct oracle IK; allow longer episodes.
        self.episode_length_s = 30.0


@configclass
class FrankaCubeLiftOracleTwistPlayRandomizedEnvCfg(FrankaCubeLiftOracleTwistPlayEnvCfg):
    """Oracle twist play with randomized cube spawn and visible lift-goal command."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.object_pose.debug_vis = True
        # Match default Isaac-Lift-Cube randomization (offset from default spawn pose).
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.1, 0.1),
            "y": (-0.25, 0.25),
            "z": (0.0, 0.0),
        }
