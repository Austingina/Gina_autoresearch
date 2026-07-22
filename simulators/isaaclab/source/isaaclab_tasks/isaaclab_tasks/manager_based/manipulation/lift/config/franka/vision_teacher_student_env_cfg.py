# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vision teacher-student env: dual cameras with GT instance mask + depth."""

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp

from .teacher_student_env_cfg import FrankaCubeLiftOracleTwistPlayEnvCfg, RewardsCfg


def _camera_cfg(
    prim_path: str,
    *,
    offset_pos: tuple[float, float, float],
    offset_rot: tuple[float, float, float, float],
    clipping_range: tuple[float, float],
) -> CameraCfg:
    return CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        height=84,
        width=84,
        data_types=["rgb", "instance_id_segmentation_fast", "distance_to_image_plane"],
        colorize_instance_id_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=clipping_range,
        ),
        offset=CameraCfg.OffsetCfg(pos=offset_pos, rot=offset_rot, convention="ros"),
    )


@configclass
class VisionObservationsCfg:
    """Dual-camera mask + depth observations (GT from Replicator)."""

    @configclass
    class PolicyCfg(ObsGroup):
        ee_depth = ObsTerm(
            func=mdp.camera_depth,
            params={"sensor_cfg": SceneEntityCfg("ee_camera"), "max_depth": 2.0},
        )
        ee_mask = ObsTerm(
            func=mdp.camera_colored_semantic_mask,
            params={"sensor_cfg": SceneEntityCfg("ee_camera")},
        )
        third_person_depth = ObsTerm(
            func=mdp.camera_depth,
            params={"sensor_cfg": SceneEntityCfg("third_person_camera"), "max_depth": 5.0},
        )
        third_person_mask = ObsTerm(
            func=mdp.camera_colored_semantic_mask,
            params={"sensor_cfg": SceneEntityCfg("third_person_camera")},
        )
        student_state = ObsTerm(func=mdp.student_proprio_observation)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class FrankaCubeLiftVisionMaskDepthPlayEnvCfg(FrankaCubeLiftOracleTwistPlayEnvCfg):
    """Play env for verifying GT mask + depth from ee + third-person cameras."""

    observations: VisionObservationsCfg = VisionObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.ee_camera = _camera_cfg(
            "{ENV_REGEX_NS}/Robot/panda_hand/ee_camera",
            offset_pos=(0.13, 0.0, -0.24),
            offset_rot=(-0.70614, 0.03701, 0.03701, -0.70614),
            clipping_range=(0.1, 2.0),
        )
        self.scene.third_person_camera = _camera_cfg(
            "{ENV_REGEX_NS}/third_person_camera",
            offset_pos=(2.2, 0.0, 1.05),
            offset_rot=(0.35355, -0.61237, -0.61237, 0.35355),
            clipping_range=(0.1, 5.0),
        )

        self.events.randomize_mask_palette = EventTerm(
            func=mdp.randomize_colored_mask_palette,
            mode="reset",
        )

        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "OFF"
        self.image_obs_list = ["ee_camera", "third_person_camera"]

        self.observations.policy.concatenate_terms = False
        self.observations.policy.enable_corruption = False


@configclass
class VisionDaggerObservationsCfg:
    """Mask + depth only (matches offline CQL ``obs_mode=vision``)."""

    @configclass
    class PolicyCfg(ObsGroup):
        ee_depth = ObsTerm(
            func=mdp.camera_depth,
            params={"sensor_cfg": SceneEntityCfg("ee_camera"), "max_depth": 2.0},
        )
        ee_mask = ObsTerm(
            func=mdp.camera_colored_semantic_mask,
            params={"sensor_cfg": SceneEntityCfg("ee_camera")},
        )
        third_person_depth = ObsTerm(
            func=mdp.camera_depth,
            params={"sensor_cfg": SceneEntityCfg("third_person_camera"), "max_depth": 5.0},
        )
        third_person_mask = ObsTerm(
            func=mdp.camera_colored_semantic_mask,
            params={"sensor_cfg": SceneEntityCfg("third_person_camera")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class VisionDaggerProprioObservationsCfg(VisionDaggerObservationsCfg):
    """Mask + depth + proprio without cube GT (``obs_mode=vision_proprio``)."""

    @configclass
    class PolicyCfg(VisionDaggerObservationsCfg.PolicyCfg):
        student_state = ObsTerm(func=mdp.student_vision_proprio_observation)

    policy: PolicyCfg = PolicyCfg()


THIRDPERSON_CAMERA_CLOSE_OFFSET = (1.15, 0.0, 0.72)
THIRDPERSON_CAMERA_FAR_OFFSET = (2.2, 0.0, 1.05)


def apply_vision_camera_settings(
    cfg,
    *,
    camera_height: int,
    camera_width: int,
    topdown_close: bool = False,
) -> None:
    """Match demo-collection / G_θ training camera resolution and top-down pose."""
    cfg.scene.ee_camera.height = camera_height
    cfg.scene.ee_camera.width = camera_width
    cfg.scene.third_person_camera.height = camera_height
    cfg.scene.third_person_camera.width = camera_width
    if topdown_close:
        cfg.scene.third_person_camera.offset.pos = THIRDPERSON_CAMERA_CLOSE_OFFSET


def _setup_vision_dagger_cameras(cfg) -> None:
    cfg.scene.ee_camera = _camera_cfg(
        "{ENV_REGEX_NS}/Robot/panda_hand/ee_camera",
        offset_pos=(0.13, 0.0, -0.24),
        offset_rot=(-0.70614, 0.03701, 0.03701, -0.70614),
        clipping_range=(0.1, 2.0),
    )
    cfg.scene.third_person_camera = _camera_cfg(
        "{ENV_REGEX_NS}/third_person_camera",
        offset_pos=(2.2, 0.0, 1.05),
        offset_rot=(0.35355, -0.61237, -0.61237, 0.35355),
        clipping_range=(0.1, 5.0),
    )
    cfg.events.randomize_mask_palette = EventTerm(
        func=mdp.randomize_colored_mask_palette,
        mode="reset",
    )
    cfg.num_rerenders_on_reset = 3
    cfg.sim.render.antialiasing_mode = "OFF"
    cfg.image_obs_list = ["ee_camera", "third_person_camera"]
    cfg.observations.policy.concatenate_terms = False
    cfg.observations.policy.enable_corruption = False


@configclass
class FrankaCubeLiftVisionTeacherStudentEnvCfg(FrankaCubeLiftOracleTwistPlayEnvCfg):
    """Vision DAgger training: IK-Rel twist actions + dual-camera mask/depth obs."""

    observations: VisionDaggerObservationsCfg = VisionDaggerObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _setup_vision_dagger_cameras(self)
        self.episode_length_s = 30.0


@configclass
class FrankaCubeLiftVisionTeacherStudentProprioEnvCfg(FrankaCubeLiftVisionTeacherStudentEnvCfg):
    """Vision DAgger with optional 7D proprio (no cube position GT)."""

    observations: VisionDaggerProprioObservationsCfg = VisionDaggerProprioObservationsCfg()
