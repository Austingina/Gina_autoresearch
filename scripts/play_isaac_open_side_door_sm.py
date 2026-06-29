"""IsaacLab Franka side-door opening state-machine demo.

This extends the official open drawer state-machine idea to the lower Sektion
cabinet side doors. It reuses the manager-based Franka IK absolute task and
retargets the cabinet frame from `drawer_handle_top` to `door_left/right_nob_link`.

Unlike the previous Arena/ManiSkill attempts, the interaction anchor is an
actual USD body in the cabinet asset:

  /World/envs/env_0/Cabinet/door_right_nob_link
  /World/envs/env_0/Cabinet/door_left_nob_link
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Open lower Sektion cabinet side door with Franka IK state machine.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--door", choices=["left", "right"], default="right")
parser.add_argument("--env_seed", type=int, default=8)
parser.add_argument("--robot_x_offset", type=float, default=-0.08, help="Move Franka base away from cabinet.")
parser.add_argument("--max_steps", type=int, default=900, help="Fixed steps for headless validation.")
parser.add_argument("--direction_frame", choices=["world", "knob", "hinge", "contact"], default="world")
parser.add_argument("--open_axis", choices=["x", "y", "z"], default="x", help="Axis to pull along.")
parser.add_argument("--open_sign", type=float, default=1.0)
parser.add_argument("--open_y", type=float, default=0.0, help="Optional pull component before frame transform.")
parser.add_argument("--approach_axis", choices=["x", "y", "z"], default="x")
parser.add_argument("--approach_sign", type=float, default=-1.0)
parser.add_argument("--approach_dist", type=float, default=0.0)
parser.add_argument("--grasp_offset", type=float, default=0.0)
parser.add_argument("--open_dist", type=float, default=0.28)
parser.add_argument("--open_duration", type=float, default=3.0)
parser.add_argument(
    "--post_pull_dist",
    type=float,
    default=-0.18,
    help="Additional finite pull after grasp/open phase; negative x matches the visible blue-arrow reverse pull.",
)
parser.add_argument("--hold_retreat_scale", type=float, default=1.0)
parser.add_argument("--position_threshold", type=float, default=0.04)
parser.add_argument("--close_threshold", type=float, default=0.08)
parser.add_argument("--use_finger_midpoint", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--frame_alignment", choices=["drawer", "identity"], default="drawer")
parser.add_argument(
    "--fixed_quat",
    type=float,
    nargs=4,
    default=[0.707, 0.0, 0.707, 0.0],
    metavar=("W", "X", "Y", "Z"),
    help="Optional fixed desired TCP quaternion in wxyz.",
)
parser.add_argument("--knob_collision_radius", type=float, default=0.016, help="Invisible grasp collider radius.")
parser.add_argument("--knob_collision_length", type=float, default=0.06, help="Invisible grasp collider length.")
parser.add_argument("--knob_collision_axis", choices=["X", "Y", "Z"], default="Z")
parser.add_argument("--show_grasp_collider", action="store_true", default=False)
parser.add_argument("--no_knob_collision", action="store_true", default=False)
parser.add_argument("--gripper_close_cmd", type=float, default=0.016)
parser.add_argument("--include_prealign", action="store_true", default=False)
parser.add_argument("--print_start_joints", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from collections.abc import Sequence

import gymnasium as gym
import torch
from isaaclab.sim.utils.prims import make_uninstanceable
from pxr import UsdGeom, UsdPhysics

from isaaclab.sensors import FrameTransformer
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.sim.schemas import CollisionPropertiesCfg, define_collision_properties
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
from isaaclab.sim.utils import bind_physics_material, clone, get_current_stage
from isaaclab.utils.math import quat_apply

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import CabinetEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


KNOB_CENTER_OFFSETS = {
    # Local offset from door_*_nob_link origin to the visual knob bbox center.
    # Measured from the loaded Sektion USD; the link origin is near the door
    # joint frame, not at the round knob.
    "right": (0.02361, -0.35178, 0.18507),
    "left": (0.02359, 0.35101, 0.18455),
}

class GripperState:
    OPEN = 0.04
    CLOSE = 0.0


ALIGNED_START_JOINT_POS = {
    # Captured after the old ALIGN_WRIST phase converged to fixed_quat.
    # This makes the task trajectory start from an already knob-parallel EE.
    "panda_joint1": 1.010539,
    "panda_joint2": -1.553489,
    "panda_joint3": -1.199929,
    "panda_joint4": -2.769814,
    "panda_joint5": -2.634661,
    "panda_joint6": 2.475978,
    "panda_joint7": -0.929505,
    "panda_finger_joint1": GripperState.OPEN,
    "panda_finger_joint2": GripperState.OPEN,
}


class OpenSideDoorState:
    REST = 0
    ALIGN_WRIST = 1
    APPROACH_INFRONT_KNOB = 2
    APPROACH_KNOB = 3
    GRASP_KNOB = 4
    OPEN_DOOR = 5
    HOLD = 6


class OpenSideDoorSm:
    """Small torch state machine that outputs IK-Abs pose + gripper command."""

    def __init__(
        self,
        dt: float,
        num_envs: int,
        device: torch.device | str,
        *,
        include_prealign: bool,
        direction_frame: str,
        open_axis: str,
        open_sign: float,
        open_y: float,
        approach_axis: str,
        approach_sign: float,
        approach_dist: float,
        grasp_offset: float,
        open_dist: float,
        open_duration: float,
        post_pull_dist: float,
        hold_retreat_scale: float,
        position_threshold: float = 0.018,
        close_threshold: float = 0.035,
        fixed_quat: tuple[float, float, float, float] | None = None,
        gripper_close_cmd: float = GripperState.CLOSE,
        gripper_action_dim: int = 1,
    ):
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.start_state = OpenSideDoorState.REST if include_prealign else OpenSideDoorState.APPROACH_INFRONT_KNOB
        self.state = torch.full((num_envs,), self.start_state, dtype=torch.long, device=device)
        self.wait = torch.zeros((num_envs,), dtype=torch.float32, device=device)
        self.position_threshold = float(position_threshold)
        self.close_threshold = float(close_threshold)
        self.direction_frame = direction_frame
        self.approach_dist = float(approach_dist)
        self.grasp_offset = float(grasp_offset)
        self.open_dist = float(open_dist)
        self.open_duration = float(open_duration)
        self.post_pull_dist = float(post_pull_dist)
        self.hold_retreat_scale = float(hold_retreat_scale)
        axis_idx = {"x": 0, "y": 1, "z": 2}[open_axis]
        self.open_vec = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
        self.open_vec[:, axis_idx] = float(open_sign)
        self.open_vec[:, 1] += float(open_y)
        self.open_vec = self.open_vec / torch.linalg.norm(self.open_vec, dim=-1, keepdim=True).clamp_min(1.0e-6)
        approach_axis_idx = {"x": 0, "y": 1, "z": 2}[approach_axis]
        self.approach_vec = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
        self.approach_vec[:, approach_axis_idx] = float(approach_sign)
        self.fixed_quat = None
        if fixed_quat is not None:
            self.fixed_quat = torch.tensor(fixed_quat, dtype=torch.float32, device=device).repeat(num_envs, 1)
        self.gripper_close_cmd = float(gripper_close_cmd)
        self.gripper_action_dim = int(gripper_action_dim)
        self.latched_pull_dir = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
        self.latched_pull_dir[:, 0] = 1.0
        self.hold_pos = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
        self.hold_quat = torch.zeros((num_envs, 4), dtype=torch.float32, device=device)
        self.hold_quat[:, 0] = 1.0
        self.open_anchor_pos = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
        self.open_anchor_dir = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
        self.open_anchor_dir[:, 0] = 1.0

    def reset_idx(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.state[env_ids] = self.start_state
        self.wait[env_ids] = 0.0
        self.latched_pull_dir[env_ids] = torch.tensor((1.0, 0.0, 0.0), dtype=torch.float32, device=self.device)
        self.hold_pos[env_ids] = 0.0
        self.hold_quat[env_ids] = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float32, device=self.device)
        self.open_anchor_pos[env_ids] = 0.0
        self.open_anchor_dir[env_ids] = torch.tensor((1.0, 0.0, 0.0), dtype=torch.float32, device=self.device)

    def _target_from_offset(
        self,
        knob_pos: torch.Tensor,
        knob_quat: torch.Tensor,
        local_offset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.direction_frame == "knob":
            return knob_pos + quat_apply(knob_quat, local_offset), knob_quat
        return knob_pos + local_offset, knob_quat

    def _direction_to_world(self, knob_quat: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        if self.direction_frame == "knob":
            return quat_apply(knob_quat, direction)
        return direction

    def compute(
        self,
        ee_pose: torch.Tensor,
        grip_center_pos: torch.Tensor,
        fingertip_pos: torch.Tensor,
        knob_pose: torch.Tensor,
        *,
        use_finger_midpoint: bool,
        hinge_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ee_pos = ee_pose[:, :3]
        knob_pos = knob_pose[:, :3]
        knob_quat = knob_pose[:, 3:7]
        control_pos = grip_center_pos if use_finger_midpoint else ee_pos
        # IK actions command the TCP. For a grasp task, target the midpoint
        # between fingertips and translate that desired point back to a TCP pose.
        tcp_from_control = ee_pos - control_pos

        desired_pos = ee_pos.clone()
        desired_quat = ee_pose[:, 3:7].clone()
        gripper = torch.full((self.num_envs,), GripperState.OPEN, device=self.device)

        contact_dir = control_pos - knob_pos
        contact_dir = contact_dir / torch.linalg.norm(contact_dir, dim=-1, keepdim=True).clamp_min(1.0e-6)
        if self.direction_frame == "contact":
            pull_dir = contact_dir * torch.sign(self.open_vec[:, 0:1])
            approach_dir = self.open_vec
        elif self.direction_frame == "hinge":
            if hinge_pos is None:
                raise ValueError("hinge_pos is required when direction_frame='hinge'")
            hinge_axis = torch.zeros_like(knob_pos)
            hinge_axis[:, 2] = 1.0
            radial = knob_pos - hinge_pos
            pull_dir = torch.cross(hinge_axis, radial, dim=-1)
            pull_dir = pull_dir * torch.sign(self.open_vec[:, 0:1])
            approach_dir = pull_dir
        else:
            pull_dir = self._direction_to_world(knob_quat, self.open_vec)
            approach_dir = pull_dir
        pull_dir = pull_dir / torch.linalg.norm(pull_dir, dim=-1, keepdim=True).clamp_min(1.0e-6)
        approach_dir = approach_dir / torch.linalg.norm(approach_dir, dim=-1, keepdim=True).clamp_min(1.0e-6)
        left_finger_pos = fingertip_pos[..., 0, :]
        right_finger_pos = fingertip_pos[..., 1, :]
        grip_dist = torch.linalg.norm(control_pos - knob_pos, dim=-1)
        left_dist = torch.linalg.norm(left_finger_pos - knob_pos, dim=-1)
        right_dist = torch.linalg.norm(right_finger_pos - knob_pos, dim=-1)
        fingers_around_knob = torch.maximum(left_dist, right_dist) < self.close_threshold
        # Approach from just outside the knob, then pull along the same current knob normal.
        approach_offset = -approach_dir * self.approach_dist
        grasp_offset = pull_dir * self.grasp_offset
        # During opening, keep chasing the moving knob plus a growing tangent/radial offset.
        open_alpha = torch.clamp((self.wait / 2.8).unsqueeze(-1), 0.0, 1.0)
        open_offset = pull_dir * (self.grasp_offset + self.open_dist * open_alpha)
        target_quat = self.fixed_quat if self.fixed_quat is not None else knob_quat

        for state_id in range(7):
            mask = self.state == state_id
            if not mask.any():
                continue
            if state_id == OpenSideDoorState.REST:
                desired_pos[mask] = ee_pos[mask]
                desired_quat[mask] = ee_pose[mask, 3:7]
                gripper[mask] = GripperState.OPEN
                if (self.wait[mask] >= 0.4).any():
                    self.state[mask] = OpenSideDoorState.ALIGN_WRIST
                    self.wait[mask] = 0.0
            elif state_id == OpenSideDoorState.ALIGN_WRIST:
                desired_pos[mask] = ee_pos[mask]
                desired_quat[mask] = target_quat[mask]
                gripper[mask] = GripperState.OPEN
                if (self.wait[mask] >= 0.6).any():
                    self.state[mask] = OpenSideDoorState.APPROACH_INFRONT_KNOB
                    self.wait[mask] = 0.0
            elif state_id == OpenSideDoorState.APPROACH_INFRONT_KNOB:
                pos = knob_pos[mask] + approach_offset[mask]
                desired_pos[mask], desired_quat[mask] = pos + tcp_from_control[mask], target_quat[mask]
                gripper[mask] = GripperState.OPEN
                control_to_pre = torch.linalg.norm(control_pos[mask] - pos, dim=-1)
                reached = control_to_pre < self.position_threshold
                touched = (grip_dist[mask] < self.close_threshold) | fingers_around_knob[mask]
                if touched.any():
                    transition_mask = mask.clone()
                    transition_mask[mask] = touched
                    gripper[transition_mask] = self.gripper_close_cmd
                    self.state[transition_mask] = OpenSideDoorState.GRASP_KNOB
                    self.wait[transition_mask] = 0.0
                elif ((reached & (self.wait[mask] >= 0.2)) | (self.wait[mask] >= 1.0)).any():
                    self.state[mask] = OpenSideDoorState.APPROACH_KNOB
                    self.wait[mask] = 0.0
            elif state_id == OpenSideDoorState.APPROACH_KNOB:
                pos = knob_pos[mask] - approach_dir[mask] * 0.01
                desired_pos[mask], desired_quat[mask] = pos + tcp_from_control[mask], target_quat[mask]
                gripper[mask] = GripperState.OPEN
                reached = (grip_dist[mask] < self.close_threshold) | fingers_around_knob[mask]
                transition = reached | (self.wait[mask] >= 0.45)
                if transition.any():
                    transition_mask = mask.clone()
                    transition_mask[mask] = transition
                    gripper[transition_mask] = self.gripper_close_cmd
                    self.state[transition_mask] = OpenSideDoorState.GRASP_KNOB
                    self.wait[transition_mask] = 0.0
            elif state_id == OpenSideDoorState.GRASP_KNOB:
                if self.direction_frame == "contact":
                    self.latched_pull_dir[mask] = contact_dir[mask]
                pos = knob_pos[mask] + grasp_offset[mask]
                desired_pos[mask], desired_quat[mask] = pos + tcp_from_control[mask], target_quat[mask]
                gripper[mask] = self.gripper_close_cmd
                if (self.wait[mask] >= 0.8).any():
                    self.open_anchor_pos[mask] = knob_pos[mask]
                    self.open_anchor_dir[mask] = pull_dir[mask]
                    self.state[mask] = OpenSideDoorState.OPEN_DOOR
                    self.wait[mask] = 0.0
            elif state_id == OpenSideDoorState.OPEN_DOOR:
                if self.direction_frame == "contact":
                    self.open_anchor_dir[mask] = self.latched_pull_dir[mask]
                pos = self.open_anchor_pos[mask] + self.open_anchor_dir[mask] * (
                    self.grasp_offset + self.open_dist * open_alpha[mask]
                )
                desired_pos[mask], desired_quat[mask] = pos + tcp_from_control[mask], target_quat[mask]
                gripper[mask] = self.gripper_close_cmd
                if (self.wait[mask] >= self.open_duration).any():
                    self.hold_pos[mask] = (
                        ee_pos[mask] + self.open_anchor_dir[mask] * self.post_pull_dist * self.hold_retreat_scale
                    )
                    self.hold_quat[mask] = target_quat[mask]
                    self.state[mask] = OpenSideDoorState.HOLD
                    self.wait[mask] = 0.0
            else:
                desired_pos[mask] = self.hold_pos[mask]
                desired_quat[mask] = self.hold_quat[mask]
                gripper[mask] = self.gripper_close_cmd

        self.wait += self.dt
        return torch.cat(
            [desired_pos, desired_quat, gripper.unsqueeze(-1).repeat(1, self.gripper_action_dim)], dim=-1
        )


def _configure_side_door_frame(env_cfg: CabinetEnvCfg, door: str, frame_alignment: str) -> str:
    knob_prim = f"{{ENV_REGEX_NS}}/Cabinet/door_{door}_nob_link"
    env_cfg.scene.cabinet_frame.target_frames[0].prim_path = knob_prim
    env_cfg.scene.cabinet_frame.target_frames[0].name = f"door_{door}_nob"
    frame_rot = (0.5, 0.5, -0.5, -0.5) if frame_alignment == "drawer" else (1.0, 0.0, 0.0, 0.0)
    env_cfg.scene.cabinet_frame.target_frames[0].offset = OffsetCfg(
        pos=KNOB_CENTER_OFFSETS[door],
        rot=frame_rot,
    )
    # Make the selected side-door task a passive articulated-object interaction.
    # The stock cabinet config was tuned for drawer RL and keeps the side doors
    # springy; that makes the small round knobs hard to open with a Franka gripper.
    env_cfg.scene.cabinet.actuators["doors"].stiffness = 0.0
    env_cfg.scene.cabinet.actuators["doors"].damping = 0.5
    return f"door_{door}_joint"


def _add_knob_collision(
    stage,
    cabinet_prim_path: str,
    door: str,
    radius: float,
    length: float,
    axis: str,
    show_grasp_collider: bool,
) -> None:
    """Add a graspable collider at the visible round knob.

    The Sektion USD exposes `door_*_nob_link/visuals`, but the corresponding
    `collisions` prim has no bounded geometry. The original knob visual stays
    visible while this collider supplies the missing physical grasp target.
    """
    if radius <= 0.0 or length <= 0.0:
        return
    offset = KNOB_CENTER_OFFSETS[door]
    make_uninstanceable(cabinet_prim_path, stage=stage)
    material_path = f"{cabinet_prim_path}/door_{door}_nob_link/collisions/ig_knob_grippy_material"
    material_cfg = RigidBodyMaterialCfg(
        static_friction=2.0,
        dynamic_friction=2.0,
        friction_combine_mode="max",
        restitution=0.0,
    )
    material_cfg.func(material_path, material_cfg)

    handle_path = f"{cabinet_prim_path}/door_{door}_nob_link/collisions/ig_knob_grasp_collider"
    handle = UsdGeom.Capsule.Define(stage, handle_path)
    handle.CreateRadiusAttr(radius)
    handle.CreateHeightAttr(length)
    handle.CreateAxisAttr(axis)
    handle.CreateDisplayColorAttr([(0.82, 0.82, 0.78)])
    handle.AddTranslateOp().Set(offset)
    if not show_grasp_collider:
        UsdGeom.Imageable(handle.GetPrim()).MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(handle.GetPrim())
    define_collision_properties(
        handle_path,
        CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.01,
            rest_offset=0.0,
            torsional_patch_radius=0.02,
            min_torsional_patch_radius=0.01,
        ),
        stage=stage,
    )
    bind_physics_material(handle_path, material_path, stage=stage)

    print(
        f"[INFO] Added knob grasp collider: path={handle_path}, offset={offset}, "
        f"radius={radius:.3f}, length={length:.3f}, axis={axis}, visible={show_grasp_collider}"
    )


@clone
def _spawn_cabinet_with_knob_collision(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    prim = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    if not args_cli.no_knob_collision:
        _add_knob_collision(
            get_current_stage(),
            prim_path,
            args_cli.door,
            args_cli.knob_collision_radius,
            args_cli.knob_collision_length,
            args_cli.knob_collision_axis,
            args_cli.show_grasp_collider,
        )
    return prim


def main():
    env_cfg: CabinetEnvCfg = parse_env_cfg(
        "Isaac-Open-Drawer-Franka-IK-Abs-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.env_seed
    env_cfg.episode_length_s = 20.0
    env_cfg.scene.robot.init_state.pos = (args_cli.robot_x_offset, 0.0, 0.0)
    if not args_cli.include_prealign:
        env_cfg.scene.robot.init_state.joint_pos = ALIGNED_START_JOINT_POS.copy()
    env_cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    env_cfg.actions.gripper_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_.*"],
        scale=1.0,
        offset=0.0,
        use_default_offset=False,
    )
    joint_name = _configure_side_door_frame(env_cfg, args_cli.door, args_cli.frame_alignment)
    env_cfg.scene.cabinet.spawn.func = _spawn_cabinet_with_knob_collision
    env = gym.make("Isaac-Open-Drawer-Franka-IK-Abs-v0", cfg=env_cfg)
    env.reset()

    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    actions[:, 7:] = GripperState.OPEN
    sm = OpenSideDoorSm(
        env_cfg.sim.dt * env_cfg.decimation,
        env.unwrapped.num_envs,
        env.unwrapped.device,
        include_prealign=args_cli.include_prealign,
        open_axis=args_cli.open_axis,
        open_sign=args_cli.open_sign,
        open_y=args_cli.open_y,
        approach_axis=args_cli.approach_axis,
        approach_sign=args_cli.approach_sign,
        direction_frame=args_cli.direction_frame,
        approach_dist=args_cli.approach_dist,
        grasp_offset=args_cli.grasp_offset,
        open_dist=args_cli.open_dist,
        open_duration=args_cli.open_duration,
        post_pull_dist=args_cli.post_pull_dist,
        hold_retreat_scale=args_cli.hold_retreat_scale,
        position_threshold=args_cli.position_threshold,
        close_threshold=args_cli.close_threshold,
        fixed_quat=tuple(args_cli.fixed_quat) if args_cli.fixed_quat is not None else None,
        gripper_close_cmd=args_cli.gripper_close_cmd,
        gripper_action_dim=env.unwrapped.action_space.shape[-1] - 7,
    )
    door_joint_id = env.unwrapped.scene["cabinet"].find_joints(joint_name)[0][0]
    door_body_name = f"door_{args_cli.door}_link"
    door_body_id = env.unwrapped.scene["cabinet"].data.body_names.index(door_body_name)
    print(
        f"[INFO] Opening {args_cli.door} side door: knob=door_{args_cli.door}_nob_link, "
        f"knob_center_offset={KNOB_CENTER_OFFSETS[args_cli.door]}, "
        f"joint={joint_name}, direction_frame={args_cli.direction_frame}, "
        f"open_axis={args_cli.open_axis}, open_sign={args_cli.open_sign}, open_y={args_cli.open_y}, "
        f"frame_alignment={args_cli.frame_alignment}, fixed_quat={args_cli.fixed_quat}"
    )

    step = 0
    printed_start_joints = False
    while simulation_app.is_running():
        with torch.inference_mode():
            dones = env.step(actions)[-2]
            ee_frame_tf: FrameTransformer = env.unwrapped.scene["ee_frame"]
            ee_pos = ee_frame_tf.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            ee_quat = ee_frame_tf.data.target_quat_w[..., 0, :].clone()
            fingertip_pos = ee_frame_tf.data.target_pos_w[..., 1:, :].clone() - env.unwrapped.scene.env_origins.unsqueeze(1)
            left_finger_pos = fingertip_pos[..., 0, :]
            right_finger_pos = fingertip_pos[..., 1, :]
            grip_center_pos = 0.5 * (left_finger_pos + right_finger_pos)

            cabinet_frame_tf: FrameTransformer = env.unwrapped.scene["cabinet_frame"]
            knob_pos = cabinet_frame_tf.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            knob_quat = cabinet_frame_tf.data.target_quat_w[..., 0, :].clone()
            hinge_pos = (
                env.unwrapped.scene["cabinet"].data.body_pos_w[:, door_body_id, :].clone()
                - env.unwrapped.scene.env_origins
            )
            actions = sm.compute(
                torch.cat([ee_pos, ee_quat], dim=-1),
                grip_center_pos,
                fingertip_pos,
                torch.cat([knob_pos, knob_quat], dim=-1),
                use_finger_midpoint=args_cli.use_finger_midpoint,
                hinge_pos=hinge_pos,
            )
            tcp_dist = torch.linalg.norm(ee_pos - knob_pos, dim=-1)
            grip_dist = torch.linalg.norm(grip_center_pos - knob_pos, dim=-1)
            left_dist = torch.linalg.norm(left_finger_pos - knob_pos, dim=-1)
            right_dist = torch.linalg.norm(right_finger_pos - knob_pos, dim=-1)
            span = right_finger_pos - left_finger_pos
            if args_cli.print_start_joints and not printed_start_joints:
                if (
                    int(sm.state[0].item()) == OpenSideDoorState.APPROACH_INFRONT_KNOB
                    and grip_dist[0].item() < 0.09
                ):
                    robot = env.unwrapped.scene["robot"]
                    joint_names = robot.data.joint_names
                    joint_pos = robot.data.joint_pos[0]
                    joint_items = [
                        f'"{name}": {joint_pos[i].item():.6f}'
                        for i, name in enumerate(joint_names)
                        if name.startswith("panda_joint") or name.startswith("panda_finger")
                    ]
                    print("[START_JOINTS] {" + ", ".join(joint_items) + "}")
                    printed_start_joints = True

            if step % 60 == 0:
                joint_pos = env.unwrapped.scene["cabinet"].data.joint_pos[:, door_joint_id]
                print(
                    f"[STEP {step:04d}] joint={joint_pos[0].item():.3f} "
                    f"tcp_to_knob={tcp_dist[0].item():.3f} grip_to_knob={grip_dist[0].item():.3f} "
                    f"lf={left_dist[0].item():.3f} rf={right_dist[0].item():.3f} "
                    f"span=({span[0,0].item():.3f},{span[0,1].item():.3f},{span[0,2].item():.3f}) "
                    f"sm={int(sm.state[0].item())}"
                )
            if dones.any():
                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                sm.reset_idx(done_ids)
                actions[done_ids] = 0.0
                actions[done_ids, 3] = 1.0
                actions[done_ids, 7:] = GripperState.OPEN
            step += 1
            if args_cli.headless and step >= args_cli.max_steps:
                break

    joint_pos = env.unwrapped.scene["cabinet"].data.joint_pos[:, door_joint_id]
    print(f"[RESULT] final_{joint_name}={joint_pos[0].item():.4f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
