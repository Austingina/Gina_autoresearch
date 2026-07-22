# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for oracle-based goal-conditioned data collection."""

import isaaclab.sim as sim_utils
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp

from . import joint_pos_env_cfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class ObservationsCfg:
    """Observation specifications for goal-conditioned offline RL data collection."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Proprioceptive state, goal, and RGB observations."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)
        ee_camera = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("ee_camera"), "data_type": "rgb", "normalize": False},
        )
        third_person_camera = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("third_person_camera"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class FrankaCubeLiftOracleDataCollectionEnvCfg(joint_pos_env_cfg.FrankaCubeLiftEnvCfg):
    """Lift-cube environment for oracle data collection with RGB observations."""

    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Use the same stiff controller + IK action interface as the official lift oracle demo.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        # Hide command / frame debug markers from camera observations.
        self.commands.object_pose.debug_vis = False
        self.scene.ee_frame.debug_vis = False

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
                # Lower and slightly closer to the finger workspace so the cube stays in view during grasp.
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
        self.image_obs_list = ["ee_camera", "third_person_camera"]
        self.observations.policy.enable_corruption = False
        self.observations.policy.concatenate_terms = False
