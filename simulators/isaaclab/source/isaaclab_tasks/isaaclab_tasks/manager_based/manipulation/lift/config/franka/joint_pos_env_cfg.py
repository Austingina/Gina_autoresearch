# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )


@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


@configclass
class MotrixFrankaCubeLiftEnvCfg(FrankaCubeLiftEnvCfg):
    """Franka cube lift task with MotrixLab-aligned reward shaping.

    The scene, random cube reset range, and randomized target command are inherited
    from Isaac Lab's built-in lift task. The reward shaping follows the MotrixLab
    task more closely so the agent first learns a robust grasp/lift behavior and
    then tracks randomized lift targets.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.contact_grasp = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_.*finger",
            update_period=0.0,
            history_length=3,
            debug_vis=False,
        )

        # Match MotrixLab's successful shaping more closely. The end-effector gates
        # are slightly wider than MotrixLab's 2 cm site distance to account for the
        # different frame definitions between MotrixSim and Isaac Lab.
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 128 * 1024

        self.rewards.reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.5)
        self.rewards.open_before_grasp = RewTerm(
            func=mdp.gripper_open_before_grasp,
            params={"approach_threshold": 0.13, "close_threshold": 0.055, "open_threshold": 0.03},
            weight=12.0,
        )
        self.rewards.closed_before_grasp = RewTerm(
            func=mdp.gripper_closed_before_grasp,
            params={"approach_threshold": 0.13, "close_threshold": 0.055, "closed_threshold": 0.012},
            weight=-12.0,
        )
        self.rewards.close_when_near_object = RewTerm(
            func=mdp.gripper_close_near_object,
            params={"ee_threshold": 0.055},
            weight=3.0,
        )
        self.rewards.object_grasped = RewTerm(
            func=mdp.object_grasped,
            params={"ee_threshold": 0.07, "force_threshold": 0.5, "min_finger_opening": 0.006},
            weight=20.0,
        )
        self.rewards.lifting_object = RewTerm(
            func=mdp.object_lifted_when_grasped,
            params={"minimal_height": 0.04, "ee_threshold": 0.07, "force_threshold": 0.5, "min_finger_opening": 0.006},
            weight=40.0,
        )
        self.rewards.object_goal_tracking = RewTerm(
            func=mdp.object_goal_tracking_sigmoid_when_grasped,
            params={
                "minimal_height": 0.04,
                "ee_threshold": 0.07,
                "force_threshold": 0.5,
                "min_finger_opening": 0.006,
                "command_name": "object_pose",
                "gain": 8.0,
                "center": 0.3,
            },
            weight=100.0,
        )
        self.rewards.object_goal_tracking_fine_grained = RewTerm(
            func=mdp.object_goal_tracking_fine_when_grasped,
            params={
                "minimal_height": 0.04,
                "ee_threshold": 0.07,
                "force_threshold": 0.5,
                "min_finger_opening": 0.006,
                "command_name": "object_pose",
                "std": 0.4,
            },
            weight=400.0,
        )
        self.rewards.object_goal_tracking_close = RewTerm(
            func=mdp.object_goal_tracking_close_when_grasped,
            params={
                "ee_threshold": 0.07,
                "force_threshold": 0.5,
                "min_finger_opening": 0.006,
                "command_name": "object_pose",
                "close_threshold": 0.2,
                "std": 0.05,
            },
            weight=300.0,
        )

        # The official curriculum raises these penalties to -1e-1 after 10k steps,
        # which dominated the failed baseline run before the policy learned to lift.
        self.rewards.action_rate.weight = -1e-4
        self.rewards.joint_vel.weight = -1e-4
        self.curriculum.action_rate = None
        self.curriculum.joint_vel = None


@configclass
class MotrixFrankaCubeLiftEnvCfg_PLAY(MotrixFrankaCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
