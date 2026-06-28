"""Visualize a successful ManiSkill cabinet task (drawer or door).

This is a deterministic oracle visualizer for the two ManiSkill cabinet tasks:

  - OpenCabinetDrawer-v1  (κ=slide / prismatic)
  - OpenCabinetDoor-v1    (κ=revolute)

ManiSkill does not ship a Fetch motion-planning solution for these mobile
manipulation tasks, so this script directly drives the target articulated joint
from closed to its task success threshold and records sensor panels. It is meant
to verify the task, observations, segmentation/depth, and success condition
quickly. The EE-action teacher can replace this oracle driver later.
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

import mani_skill  # noqa: F401  (registers envs)


SUPPORTED_ENVS = ("OpenCabinetDrawer-v1", "OpenCabinetDoor-v1")


def _to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    valid = depth > 0
    out = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if valid.any():
        lo = float(np.percentile(depth[valid], 2))
        hi = float(np.percentile(depth[valid], 98))
        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((depth.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
        gray = (norm * 255).astype(np.uint8)
        out = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out


def _colorize_seg(seg: np.ndarray) -> np.ndarray:
    seg = np.asarray(seg)
    if seg.ndim == 3 and seg.shape[-1] == 1:
        seg = seg[..., 0]
    seg_i = seg.astype(np.int64)
    # Deterministic pseudo-random colors per instance id.
    r = (37 * seg_i + 17) % 255
    g = (67 * seg_i + 29) % 255
    b = (97 * seg_i + 43) % 255
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    rgb[seg_i == 0] = 0
    return rgb


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _camera_panel(obs: dict, camera_name: str) -> np.ndarray:
    data = obs["sensor_data"][camera_name]
    rgb = _to_numpy(data["rgb"])[0]
    depth = _normalize_depth(_to_numpy(data["depth"])[0])
    seg = _colorize_seg(_to_numpy(data["segmentation"])[0])
    return np.concatenate(
        [
            _label(rgb, f"{camera_name} rgb"),
            _label(depth, f"{camera_name} depth"),
            _label(seg, f"{camera_name} seg"),
        ],
        axis=1,
    )


def _make_panel(obs: dict, env_id: str, step: int, open_frac: float, success: bool) -> np.ndarray:
    cams = list(obs["sensor_data"].keys())
    rows = [_camera_panel(obs, c) for c in cams]
    panel = np.concatenate(rows, axis=0)
    title = f"{env_id}  step={step}  open_frac={open_frac:.2f}  success={success}"
    cv2.putText(panel, title, (8, panel.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return panel


def run(env_id: str, out_dir: Path, steps: int, fps: int, seed: int) -> bool:
    if env_id not in SUPPORTED_ENVS:
        raise ValueError(f"env_id must be one of {SUPPORTED_ENVS}, got {env_id!r}")

    out_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(env_id, num_envs=1, obs_mode="rgb+depth+segmentation", sim_backend="physx_cuda")
    base = env.unwrapped
    obs, info = env.reset(seed=seed)

    qmin = base.handle_link.joint.limits[:, 0].reshape(1, 1)
    target = base.target_qpos.reshape(1, 1)
    # Add a small margin beyond the threshold while staying below qmax.
    qmax = base.handle_link.joint.limits[:, 1].reshape(1, 1)
    final_q = torch.minimum(target + 0.05 * (qmax - qmin), qmax)
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    frames: list[np.ndarray] = []

    success = False
    for t in range(steps):
        alpha = min(1.0, t / max(steps - 12, 1))
        # Smoothstep makes the visual motion less abrupt.
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        q = qmin + alpha * (final_q - qmin)
        base.cabinet.set_qpos(q)
        base.cabinet.set_qvel(torch.zeros_like(q))
        if base.gpu_sim_enabled:
            base.scene._gpu_apply_all()
            base.scene.px.gpu_update_articulation_kinematics()
            base.scene._gpu_fetch_all()
        obs, _, _, _, info = env.step(zero_action)
        open_frac = float((base.handle_link.joint.qpos[0] - qmin[0, 0]) / torch.clamp(qmax[0, 0] - qmin[0, 0], min=1e-6))
        success = bool(_to_numpy(info["success"])[0])
        frames.append(_make_panel(obs, env_id, t, open_frac, success))

    # Hold final pose for static-success condition and clearer video ending.
    for t in range(12):
        obs, _, _, _, info = env.step(zero_action)
        success = bool(_to_numpy(info["success"])[0])
        open_frac = float((base.handle_link.joint.qpos[0] - qmin[0, 0]) / torch.clamp(qmax[0, 0] - qmin[0, 0], min=1e-6))
        frames.append(_make_panel(obs, env_id, steps + t, open_frac, success))

    video_path = out_dir / f"{env_id}_success_panel.mp4"
    imageio.mimsave(video_path, frames, fps=fps)
    imageio.imwrite(out_dir / f"{env_id}_final_panel.png", frames[-1])
    env.close()

    print(f"env_id={env_id}")
    print(f"success={success}")
    print(f"video={video_path}")
    print(f"final_panel={out_dir / f'{env_id}_final_panel.png'}")
    return success


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", choices=SUPPORTED_ENVS, required=True)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path("runs") / f"maniskill_{args.env_id}_success_{ts}"
    ok = run(args.env_id, out_dir, steps=args.steps, fps=args.fps, seed=args.seed)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
