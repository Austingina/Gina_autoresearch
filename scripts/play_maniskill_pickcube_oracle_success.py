"""Visualize a successful ManiSkill PickCube-v1 episode.

This deterministic oracle visualizer moves the cube from its initial pose to
the environment's goal position and records RGB/depth/segmentation panels. It
is a fast success/observation sanity check; the slower official Panda motion
planner can be used later for true robot-action trajectories.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

import mani_skill  # noqa: F401
from mani_skill.utils.structs import Pose


def _to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    valid = depth > 0
    out = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if valid.any():
        lo = float(np.percentile(depth[valid], 2))
        hi = float(np.percentile(depth[valid], 98))
        if hi <= lo:
            hi = lo + 1.0
        gray = (np.clip((depth.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)
        out = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out


def _colorize_seg(seg: np.ndarray) -> np.ndarray:
    if seg.ndim == 3 and seg.shape[-1] == 1:
        seg = seg[..., 0]
    seg_i = seg.astype(np.int64)
    rgb = np.stack([(37 * seg_i + 17) % 255, (67 * seg_i + 29) % 255, (97 * seg_i + 43) % 255], axis=-1)
    rgb = rgb.astype(np.uint8)
    rgb[seg_i == 0] = 0
    return rgb


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _make_panel(obs: dict, step: int, success: bool) -> np.ndarray:
    # PickCube default sensor is base_camera.
    camera_name = next(iter(obs["sensor_data"].keys()))
    data = obs["sensor_data"][camera_name]
    rgb = _to_numpy(data["rgb"])[0]
    depth = _normalize_depth(_to_numpy(data["depth"])[0])
    seg = _colorize_seg(_to_numpy(data["segmentation"])[0])
    panel = np.concatenate(
        [_label(rgb, "base_camera rgb"), _label(depth, "base_camera depth"), _label(seg, "base_camera seg")],
        axis=1,
    )
    title = f"PickCube-v1  step={step}  success={success}"
    cv2.putText(panel, title, (8, panel.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return panel


def run(out_dir: Path, steps: int, fps: int, seed: int) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make("PickCube-v1", num_envs=1, obs_mode="rgb+depth+segmentation", sim_backend="physx_cuda")
    base = env.unwrapped
    obs, _ = env.reset(seed=seed)
    raw0 = base.cube.pose.raw_pose.clone()
    goal = obs["extra"]["goal_pos"].clone()
    raw_goal = raw0.clone()
    raw_goal[:, 0:3] = goal
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)

    frames: list[np.ndarray] = []
    success = False
    for t in range(steps):
        alpha = min(1.0, t / max(steps - 12, 1))
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        raw = raw0 * (1.0 - alpha) + raw_goal * alpha
        raw[:, 3:7] = raw0[:, 3:7]
        base.cube.set_pose(Pose.create_from_pq(p=raw[:, 0:3], q=raw[:, 3:7]))
        base.cube.set_linear_velocity(torch.zeros((1, 3), device=goal.device))
        base.cube.set_angular_velocity(torch.zeros((1, 3), device=goal.device))
        if base.gpu_sim_enabled:
            base.scene._gpu_apply_all()
            base.scene.px.gpu_update_articulation_kinematics()
            base.scene._gpu_fetch_all()
        obs, _, _, _, info = env.step(zero_action)
        success = bool(_to_numpy(info["success"])[0])
        frames.append(_make_panel(obs, t, success))

    for t in range(12):
        obs, _, _, _, info = env.step(zero_action)
        success = bool(_to_numpy(info["success"])[0])
        frames.append(_make_panel(obs, steps + t, success))

    video_path = out_dir / "PickCube-v1_success_panel.mp4"
    final_path = out_dir / "PickCube-v1_final_panel.png"
    imageio.mimsave(video_path, frames, fps=fps)
    imageio.imwrite(final_path, frames[-1])
    env.close()

    print("env_id=PickCube-v1")
    print(f"success={success}")
    print(f"video={video_path}")
    print(f"final_panel={final_path}")
    return success


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path("runs") / f"maniskill_PickCube-v1_success_{ts}"
    ok = run(out_dir, steps=args.steps, fps=args.fps, seed=args.seed)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
