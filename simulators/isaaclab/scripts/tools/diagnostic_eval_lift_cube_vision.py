# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Headless diagnostic eval for vision DAgger student — saves per-episode trajectories + failure modes.

Example:
  ./isaaclab.sh -p scripts/tools/diagnostic_eval_lift_cube_vision.py \\
    --checkpoint /path/to/student_best.pt \\
    --output_dir /path/to/eval_diagnostics \\
    --headless --num_episodes 20
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
from datetime import datetime, timezone

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diagnostic eval for vision DAgger lift student.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-Vision-Teacher-Student-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_episodes", type=int, default=20)
parser.add_argument("--image_size", type=int, default=64)
parser.add_argument("--camera_height", type=int, default=0)
parser.add_argument("--camera_width", type=int, default=0)
parser.add_argument("--topdown_close", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--position_threshold", type=float, default=0.02)
parser.add_argument("--arm_scale", type=float, default=0.5)
parser.add_argument("--episode_length_s", type=float, default=25.0)
parser.add_argument("--trajectory_stride", type=int, default=10)
parser.add_argument("--export_field_hdf5", type=str, default="")
parser.add_argument("--export_field_stride", type=int, default=25)
parser.add_argument("--fixed_lift_goal", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--lift_goal_x", type=float, default=0.5)
parser.add_argument("--lift_goal_y", type=float, default=0.0)
parser.add_argument("--lift_goal_z", type=float, default=0.375)
parser.add_argument("--randomize_cube", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--cube_x_offset", type=float, default=None)
parser.add_argument("--cube_y_offset", type=float, default=None)
parser.add_argument("--cube_z_offset", type=float, default=0.0)
parser.add_argument("--randomize_lift_goal", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--lift_goal_above_cube", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--lift_height_above_cube", type=float, default=0.20)
parser.add_argument(
    "--guidance_mode",
    type=str,
    default="anchor_only",
    choices=("anchor_only", "anchor_field_demo", "anchor_field_rule"),
)
parser.add_argument("--field_checkpoint", type=str, default="")
parser.add_argument("--autoresearch_root", type=str, default=os.environ.get("AUTORESEARCH_ROOT", ""))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import h5py
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift.mdp.data_collection import (
    EpisodeLiftSuccessTracker,
    compute_cube_to_goal_distance,
    compute_grasp_success,
    get_ee_pose_in_robot_root_frame,
    get_lift_goal_position_in_robot_root_frame,
    get_mean_finger_opening,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.observations import object_position_in_robot_root_frame
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student import (
    apply_lift_teacher_student_env_randomization,
    extract_proprio_from_policy_obs_torch,
    get_gripper_action_dim,
    infer_teacher_phase,
    reset_teacher_phase_state,
    twist_to_env_action,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.vision_policy_obs import (
    stack_policy_visual_torch,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.lift_episode_diagnostics import (
    EpisodeDiagnosticRecorder,
    aggregate_diagnostics,
)
from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.vision_networks import (
    StudentTwistPolicyBinaryGripperVision,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.vision_teacher_student_env_cfg import (
    apply_vision_camera_settings,
)


def _load_policy(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    visual_channels = StudentTwistPolicyBinaryGripperVision.resolve_visual_channels_from_checkpoint(ckpt)
    enable_aux_delta_head = StudentTwistPolicyBinaryGripperVision.resolve_aux_delta_head_from_checkpoint(ckpt)
    proprio_dim = int(ckpt.get("proprio_dim", 0))
    image_size = int(ckpt.get("image_size", args_cli.image_size))
    policy = StudentTwistPolicyBinaryGripperVision(
        visual_channels=visual_channels,
        proprio_dim=proprio_dim,
        enable_aux_delta_head=enable_aux_delta_head,
    ).to(device)
    StudentTwistPolicyBinaryGripperVision.load_from_checkpoint(
        policy, ckpt, visual_channels=visual_channels, proprio_dim=proprio_dim
    )
    policy.eval()
    return policy, ckpt, image_size


def _ee_cube_distance(env, env_id: int) -> float:
    obj = env.scene["object"].data.root_pos_w[env_id, :3]
    ee = env.scene["ee_frame"].data.target_pos_w[env_id, 0, :3]
    return float(torch.norm(obj - ee).item())


def _obs_frame_np(obs: dict[str, torch.Tensor], key: str, env_id: int) -> np.ndarray:
    return obs[key][env_id].detach().cpu().numpy().astype(np.float32)


def _new_field_buffer() -> dict[str, list[np.ndarray | float]]:
    return {
        "ee_mask": [],
        "ee_depth": [],
        "third_person_mask": [],
        "third_person_depth": [],
        "student_state": [],
        "phase": [],
    }


def _append_field_buffer(
    buffer: dict[str, list[np.ndarray | float]],
    *,
    obs: dict[str, torch.Tensor],
    env_id: int,
    student_state: np.ndarray,
    phase: float,
) -> None:
    for key in ("ee_mask", "ee_depth", "third_person_mask", "third_person_depth"):
        buffer[key].append(_obs_frame_np(obs, key, env_id))
    buffer["student_state"].append(student_state.astype(np.float32))
    buffer["phase"].append(float(phase))


def _write_field_demo(
    h5_file: h5py.File | None,
    *,
    episode_id: int,
    buffer: dict[str, list[np.ndarray | float]],
) -> None:
    if h5_file is None or len(buffer["phase"]) < 2:
        return
    group = h5_file.create_group(f"data/demo_{episode_id}")
    group.attrs["success"] = True
    group.attrs["source"] = "student_state_relabel"
    obs_group = group.create_group("obs")
    for key in ("ee_mask", "ee_depth", "third_person_mask", "third_person_depth", "student_state"):
        obs_group.create_dataset(key, data=np.asarray(buffer[key], dtype=np.float32), compression="gzip", compression_opts=1)
    goals_group = group.create_group("goals")
    goals_group.create_dataset("phase", data=np.asarray(buffer["phase"], dtype=np.float32), compression="gzip", compression_opts=1)


def main():
    os.makedirs(args_cli.output_dir, exist_ok=True)
    device = args_cli.device

    guidance_field = None
    if args_cli.guidance_mode == "anchor_field_demo":
        if not args_cli.field_checkpoint:
            raise ValueError("--field_checkpoint required for anchor_field_demo")
        if args_cli.autoresearch_root:
            import sys

            sys.path.insert(0, args_cli.autoresearch_root)
        from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.guidance_field_bridge import (
            LearnedGuidanceField,
        )

        guidance_field = LearnedGuidanceField(args_cli.field_checkpoint, device=device)
    elif args_cli.guidance_mode == "anchor_field_rule":
        from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.guidance_field_bridge import (
            OracleRuleGuidanceField,
        )

        guidance_field = OracleRuleGuidanceField(device=device)

    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.episode_length_s = args_cli.episode_length_s
    cube_pose_range = None
    fixed_cube_offset = args_cli.cube_x_offset is not None or args_cli.cube_y_offset is not None
    if fixed_cube_offset:
        cube_pose_range = {
            "x": (float(args_cli.cube_x_offset or 0.0), float(args_cli.cube_x_offset or 0.0)),
            "y": (float(args_cli.cube_y_offset or 0.0), float(args_cli.cube_y_offset or 0.0)),
            "z": (float(args_cli.cube_z_offset), float(args_cli.cube_z_offset)),
        }
    apply_lift_teacher_student_env_randomization(
        env_cfg,
        randomize_cube=args_cli.randomize_cube or fixed_cube_offset,
        randomize_lift_goal=args_cli.randomize_lift_goal,
        fixed_lift_goal=args_cli.fixed_lift_goal and not args_cli.randomize_lift_goal,
        lift_goal_x=args_cli.lift_goal_x,
        lift_goal_y=args_cli.lift_goal_y,
        lift_goal_z=args_cli.lift_goal_z,
        lift_goal_above_cube=args_cli.lift_goal_above_cube,
        lift_height_above_cube=args_cli.lift_height_above_cube,
        cube_pose_range=cube_pose_range,
    )
    cam_h = args_cli.camera_height or args_cli.image_size
    cam_w = args_cli.camera_width or args_cli.image_size
    apply_vision_camera_settings(
        env_cfg,
        camera_height=cam_h,
        camera_width=cam_w,
        topdown_close=args_cli.topdown_close,
    )
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    control_dt = env_cfg.sim.dt * env_cfg.decimation
    gripper_dim = get_gripper_action_dim(env)

    policy, ckpt, image_size = _load_policy(args_cli.checkpoint, device)
    num_envs = env.num_envs
    rnn_hidden = policy.init_hidden(num_envs, device)
    ep_step = torch.zeros(num_envs, dtype=torch.long, device=device)
    success_tracker = EpisodeLiftSuccessTracker(num_envs, device)
    recorders = [EpisodeDiagnosticRecorder() for _ in range(num_envs)]
    export_h5 = h5py.File(args_cli.export_field_hdf5, "w") if args_cli.export_field_hdf5 else None
    if export_h5 is not None:
        export_h5.create_group("data")
        export_h5.attrs["checkpoint"] = args_cli.checkpoint
        export_h5.attrs["guidance_mode"] = args_cli.guidance_mode
        export_h5.attrs["field_checkpoint"] = args_cli.field_checkpoint
        export_h5.attrs["export_field_stride"] = int(args_cli.export_field_stride)
    field_buffers = [_new_field_buffer() for _ in range(num_envs)]

    env.reset()
    episodes_done = 0
    episode_results = []

    while episodes_done < args_cli.num_episodes and simulation_app.is_running():
        with torch.no_grad():
            student_obs = env.observation_manager.compute_group("policy")
            visual = stack_policy_visual_torch(
                student_obs,
                image_size=image_size,
                visual_layout=str(ckpt.get("visual_layout", "channels")),
            )
            if guidance_field is not None:
                teacher_phase = infer_teacher_phase(env)
                visual = guidance_field.augment_visual(
                    visual,
                    student_obs,
                    image_size=image_size,
                    env=env,
                    teacher_phase=teacher_phase,
                )
            visual_step = visual.unsqueeze(1)
            proprio = extract_proprio_from_policy_obs_torch(student_obs)
            proprio_step = None if proprio is None else proprio.unsqueeze(1)
            actions, rnn_hidden, _ = policy.act(visual_step, rnn_hidden, proprio_step, deterministic=True)
            twist = actions

        current_opening = get_mean_finger_opening(env)
        pre_dist = compute_cube_to_goal_distance(env)
        pre_z = env.scene["object"].data.root_pos_w[:, 2].clone()
        ee_pos_b_pre, _ = get_ee_pose_in_robot_root_frame(env)
        cube_pos_b_pre = object_position_in_robot_root_frame(env)
        goal_pos_b_pre = get_lift_goal_position_in_robot_root_frame(env)
        teacher_phase_pre = infer_teacher_phase(env)
        if export_h5 is not None:
            export_stride = max(int(args_cli.export_field_stride), 1)
            for env_id in range(num_envs):
                if int(ep_step[env_id].item()) % export_stride != 0:
                    continue
                student_state = np.zeros(10, dtype=np.float32)
                student_state[0:3] = ee_pos_b_pre[env_id].detach().cpu().numpy()
                student_state[3] = float(current_opening[env_id].item())
                student_state[4:7] = cube_pos_b_pre[env_id].detach().cpu().numpy()
                student_state[7:10] = goal_pos_b_pre[env_id].detach().cpu().numpy()
                phase = 1.0 if int(teacher_phase_pre[env_id].item()) >= 3 else 0.0
                _append_field_buffer(
                    field_buffers[env_id],
                    obs=student_obs,
                    env_id=env_id,
                    student_state=student_state,
                    phase=phase,
                )
        env_actions = twist_to_env_action(
            twist,
            control_dt=control_dt,
            current_opening=current_opening,
            arm_scale=args_cli.arm_scale,
            num_gripper_actions=gripper_dim,
            binary_gripper=True,
        )
        _, _, terminated, truncated, _ = env.step(env_actions)
        ep_step += 1

        success_early = success_tracker.on_step(
            env,
            terminated=terminated,
            truncated=truncated,
            pre_cube_to_goal=pre_dist,
            pre_cube_z_w=pre_z,
            lift_threshold=args_cli.position_threshold,
        )

        ee_pos_b, _ = get_ee_pose_in_robot_root_frame(env)
        cube_pos_b = object_position_in_robot_root_frame(env)
        goal_pos_b = get_lift_goal_position_in_robot_root_frame(env)
        teacher_phase = infer_teacher_phase(env)
        grasp_now = compute_grasp_success(env)
        lift_now = success_tracker.ever_lift_success.clone()

        for env_id in range(num_envs):
            recorders[env_id].record_step(
                step=int(ep_step[env_id].item()),
                ee_pos_b=ee_pos_b[env_id].detach().cpu().numpy(),
                cube_pos_b=cube_pos_b[env_id].detach().cpu().numpy(),
                goal_pos_b=goal_pos_b[env_id].detach().cpu().numpy(),
                dist_to_goal=float(pre_dist[env_id].item()),
                ee_cube_dist=_ee_cube_distance(env, env_id),
                gripper_open=float(current_opening[env_id].item()),
                gripper_cmd=float(twist[env_id, 6].item()),
                teacher_phase=int(teacher_phase[env_id].item()),
                grasp_now=bool(grasp_now[env_id].item()),
                lift_now=bool(lift_now[env_id].item()),
            )

        dones = terminated | truncated | success_early
        if dones.any():
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.ndim == 0:
                done_ids = done_ids.unsqueeze(0)
            for env_id in done_ids.tolist():
                if episodes_done >= args_cli.num_episodes:
                    break
                if success_tracker.ever_lift_success[env_id].item():
                    end_reason = "success"
                elif truncated[env_id].item():
                    end_reason = "timeout"
                elif terminated[env_id].item():
                    end_reason = "terminated"
                else:
                    end_reason = "done"
                diag = recorders[env_id].finalize(
                    episode_id=episodes_done,
                    end_reason=end_reason,
                    ever_grasp_flag=bool(success_tracker.ever_grasp_success[env_id].item()),
                    ever_lift_flag=bool(success_tracker.ever_lift_success[env_id].item()),
                    trajectory_stride=args_cli.trajectory_stride,
                )
                episode_results.append(diag.to_dict())
                _write_field_demo(export_h5, episode_id=episodes_done, buffer=field_buffers[env_id])
                episodes_done += 1
                recorders[env_id].reset()
                field_buffers[env_id] = _new_field_buffer()

            policy.reset_hidden(rnn_hidden, done_ids)
            reset_teacher_phase_state(env, done_ids.tolist())
            success_tracker.reset(done_ids)
            ep_step[done_ids] = 0
            if success_early.any():
                env.reset(env_ids=success_early.nonzero(as_tuple=False).squeeze(-1))

    if export_h5 is not None:
        export_h5.flush()
        export_h5.close()
    env.close()

    from isaaclab_tasks.manager_based.manipulation.lift.mdp.teacher_student.lift_episode_diagnostics import (
        EpisodeDiagnostic,
    )

    ep_objs = [EpisodeDiagnostic(**e) for e in episode_results]
    summary = aggregate_diagnostics(ep_objs)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": os.path.abspath(args_cli.checkpoint),
        "guidance_mode": args_cli.guidance_mode,
        "field_checkpoint": args_cli.field_checkpoint or None,
        "task": args_cli.task,
        "num_episodes": args_cli.num_episodes,
        "checkpoint_meta": {
            k: ckpt.get(k)
            for k in ("iteration", "success_rate", "visual_channels", "obs_mode", "policy_type")
        },
        "aggregate": summary,
        "episodes": episode_results,
    }

    out_json = os.path.join(args_cli.output_dir, "eval_diagnostics.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Wrote {out_json}")
    print(f"[OK] success_rate={summary['success_rate']:.3f} success@5cm={summary['success_rate_5cm']:.3f} modes={summary['failure_mode_counts']}")
    print(f"[OK] mean_post_grasp_wander_m={summary['mean_post_grasp_wander_m']:.4f}")


if __name__ == "__main__":
    main()
    simulation_app.close()
