# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Visualize oracle goal-conditioned lift-cube HDF5 demonstrations.

Overlays a top-down robot-base view on the camera feed with:
- red dot: current phase goal position
- arrow: end-effector linear velocity direction (from recorded 6D twist action)

Example:

.. code-block:: bash

    python scripts/tools/visualize_oracle_goal_conditioned_demos.py \\
        --dataset_file ./datasets/lift_cube_gc_oracle.hdf5 \\
        --output_dir ./runs/lift_cube_gc_viz \\
        --demo_idx 0

    ffplay runs/lift_cube_gc_viz/demo_0_preview.mp4
"""

from __future__ import annotations

import argparse
import os
import re

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize oracle goal-conditioned lift-cube HDF5 demos.")
    parser.add_argument(
        "--dataset_file",
        type=str,
        required=True,
        help="Path to the HDF5 dataset recorded by record_oracle_goal_conditioned_demos.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./runs/lift_cube_gc_viz",
        help="Directory for preview images and mp4 output.",
    )
    parser.add_argument(
        "--demo_idx",
        type=int,
        default=0,
        help="Index of the demonstration to visualize.",
    )
    parser.add_argument(
        "--camera_key",
        type=str,
        default="third_person_camera",
        help="Camera key under obs/ to display. Options: third_person_camera, ee_camera.",
    )
    parser.add_argument(
        "--framerate",
        type=int,
        default=20,
        help="Output video framerate.",
    )
    parser.add_argument(
        "--velocity_scale",
        type=float,
        default=0.25,
        help="Arrow length scale for unit linear velocity (m/s -> plot meters).",
    )
    parser.add_argument(
        "--plot_size",
        type=int,
        default=320,
        help="Side panel size in pixels (width and height).",
    )
    parser.add_argument(
        "--display_scale",
        type=int,
        default=2,
        help="Upscale factor for the final composite frame.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Maximum number of steps to render. 0 means render the full episode.",
    )
    parser.add_argument(
        "--show_opencv",
        action="store_true",
        default=False,
        help="Show a live OpenCV window while rendering (requires GUI-enabled OpenCV).",
    )
    return parser.parse_args()


def _demo_sort_key(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else 0


def list_demos(hdf5_file: h5py.File) -> list[str]:
    return sorted(hdf5_file["data"].keys(), key=_demo_sort_key)


def _read_dataset(group: h5py.Group, path: str) -> np.ndarray:
    node = group
    for part in path.split("/"):
        if part not in node:
            raise KeyError(f"Missing dataset path '{path}' in demo group.")
        node = node[part]
    return np.asarray(node)


def _to_rgb_image(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame)
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255.0).clip(0, 255).astype(np.uint8)
        else:
            image = image.clip(0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    return image


def load_demo(hdf5_path: str, demo_idx: int, camera_key: str) -> dict[str, np.ndarray]:
    with h5py.File(hdf5_path, "r") as hdf5_file:
        demo_names = list_demos(hdf5_file)
        if demo_idx < 0 or demo_idx >= len(demo_names):
            raise IndexError(f"demo_idx {demo_idx} out of range. Dataset has {len(demo_names)} demos.")
        demo_name = demo_names[demo_idx]
        demo_group = hdf5_file["data"][demo_name]

        actions = _read_dataset(demo_group, "actions")
        ee_pose = _read_dataset(demo_group, "ee_pose")
        goal_position = _read_dataset(demo_group, "goals/position")

        data = {
            "demo_name": demo_name,
            "actions": actions,
            "ee_pose": ee_pose,
            "goal_position": goal_position,
        }

        optional_paths = {
            "goal_phase": "goals/phase",
            "reward_grasp": "rewards/grasp",
            "reward_lift": "rewards/lift",
            f"camera_{camera_key}": f"obs/{camera_key}",
        }
        for key, path in optional_paths.items():
            try:
                data[key] = _read_dataset(demo_group, path)
            except KeyError:
                pass

        return data


def _setup_topdown_axes(ax: plt.Axes) -> None:
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(-0.35, 0.35)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (robot base)")
    ax.set_ylabel("y (robot base)")
    ax.grid(True, alpha=0.3)
    ax.axhline(0.0, color="0.7", linewidth=0.8)
    ax.axvline(0.5, color="0.7", linewidth=0.8)


def _action_lin_vel(action: np.ndarray) -> np.ndarray:
    return np.asarray(action[:3], dtype=np.float64)


def _action_gripper_rate(action: np.ndarray) -> float | None:
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape[0] >= 7:
        return float(action[6])
    return None


def render_topdown_panel(
    ee_pos: np.ndarray,
    goal_pos: np.ndarray,
    lin_vel: np.ndarray,
    *,
    gripper_rate: float | None = None,
    goal_phase: float | None,
    step_idx: int,
    num_steps: int,
    velocity_scale: float,
    fig_size_px: int,
) -> np.ndarray:
    """Render top-down view with red goal dot and velocity arrow."""
    dpi = 100
    fig_size_in = fig_size_px / dpi
    fig, ax = plt.subplots(figsize=(fig_size_in, fig_size_in), dpi=dpi)
    _setup_topdown_axes(ax)

    ax.plot(ee_pos[0], ee_pos[1], "o", color="#1f77b4", markersize=8, label="EE")
    ax.plot(goal_pos[0], goal_pos[1], "o", color="red", markersize=10, label="goal")

    speed = float(np.linalg.norm(lin_vel))
    if speed > 1e-6:
        direction = lin_vel / speed
        arrow_len = speed * velocity_scale
        arrow = FancyArrowPatch(
            (ee_pos[0], ee_pos[1]),
            (ee_pos[0] + direction[0] * arrow_len, ee_pos[1] + direction[1] * arrow_len),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=2.5,
            color="#00bcd4",
        )
        ax.add_patch(arrow)

    phase_text = "lift" if goal_phase is not None and goal_phase >= 0.5 else "grasp"
    info_lines = [f"|v|={speed:.3f} m/s", f"vz={lin_vel[2]:+.3f}"]
    if gripper_rate is not None:
        info_lines.append(f"gripper_rate={gripper_rate:+.4f} m/s")
    ax.set_title(f"step {step_idx + 1}/{num_steps} | phase: {phase_text}", fontsize=10)
    ax.text(
        0.02,
        0.98,
        "\n".join(info_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())
    image = buffer[..., :3].copy()
    plt.close(fig)
    return image


def compose_frame(
    camera_rgb: np.ndarray | None,
    topdown_rgb: np.ndarray,
    *,
    reward_grasp: float | None,
    reward_lift: float | None,
    display_scale: int,
) -> np.ndarray:
    camera_rgb = _to_rgb_image(camera_rgb) if camera_rgb is not None else np.zeros_like(topdown_rgb)

    cam_h, cam_w = camera_rgb.shape[:2]
    panel = cv2.resize(topdown_rgb, (cam_h, cam_h), interpolation=cv2.INTER_AREA)
    composite = np.concatenate([camera_rgb, panel], axis=1)

    status = []
    if reward_grasp is not None and reward_grasp > 0.5:
        status.append("grasp=1")
    if reward_lift is not None and reward_lift > 0.5:
        status.append("lift=1")
    if status:
        cv2.putText(
            composite,
            " | ".join(status),
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 220, 40),
            2,
            cv2.LINE_AA,
        )

    if display_scale > 1:
        composite = cv2.resize(
            composite,
            (composite.shape[1] * display_scale, composite.shape[0] * display_scale),
            interpolation=cv2.INTER_NEAREST,
        )
    return cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)


def render_demo_video(args: argparse.Namespace) -> str:
    demo = load_demo(args.dataset_file, args.demo_idx, args.camera_key)
    num_steps = len(demo["actions"])
    if args.max_steps > 0:
        num_steps = min(num_steps, args.max_steps)

    os.makedirs(args.output_dir, exist_ok=True)
    demo_name = demo["demo_name"]
    output_mp4 = os.path.join(args.output_dir, f"{demo_name}_preview.mp4")
    output_latest = os.path.join(args.output_dir, "preview_latest.png")

    camera_key = f"camera_{args.camera_key}"
    has_camera = camera_key in demo

    first_frame = compose_frame(
        demo[camera_key][0] if has_camera else None,
        render_topdown_panel(
            demo["ee_pose"][0, :3],
            demo["goal_position"][0],
            _action_lin_vel(demo["actions"][0]),
            gripper_rate=_action_gripper_rate(demo["actions"][0]),
            goal_phase=_scalar(demo.get("goal_phase"), 0, 0),
            step_idx=0,
            num_steps=num_steps,
            velocity_scale=args.velocity_scale,
            fig_size_px=args.plot_size,
        ),
        reward_grasp=_scalar(demo.get("reward_grasp"), 0, None),
        reward_lift=_scalar(demo.get("reward_lift"), 0, None),
        display_scale=args.display_scale,
    )
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        output_mp4,
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.framerate,
        (width, height),
    )

    if args.show_opencv:
        window_name = f"oracle demo {demo_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for step_idx in range(num_steps):
        goal_phase = _scalar(demo.get("goal_phase"), step_idx, None)
        reward_grasp = _scalar(demo.get("reward_grasp"), step_idx, None)
        reward_lift = _scalar(demo.get("reward_lift"), step_idx, None)
        camera = demo[camera_key][step_idx] if has_camera else None

        topdown = render_topdown_panel(
            demo["ee_pose"][step_idx, :3],
            demo["goal_position"][step_idx],
            _action_lin_vel(demo["actions"][step_idx]),
            gripper_rate=_action_gripper_rate(demo["actions"][step_idx]),
            goal_phase=goal_phase,
            step_idx=step_idx,
            num_steps=num_steps,
            velocity_scale=args.velocity_scale,
            fig_size_px=args.plot_size,
        )
        frame = compose_frame(
            camera,
            topdown,
            reward_grasp=reward_grasp,
            reward_lift=reward_lift,
            display_scale=args.display_scale,
        )
        writer.write(frame)
        cv2.imwrite(output_latest, frame)

        if args.show_opencv:
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(int(1000 / args.framerate)) & 0xFF
            if key == ord("q"):
                break

    writer.release()
    if args.show_opencv:
        cv2.destroyAllWindows()

    return output_mp4


def _scalar(array: np.ndarray | None, index: int, default: float | None) -> float | None:
    if array is None:
        return default
    value = np.asarray(array[index]).reshape(-1)[0]
    return float(value)


if __name__ == "__main__":
    args = parse_args()
    output_path = render_demo_video(args)
    print(f"[INFO] Saved visualization to: {output_path}")
    print(f"[INFO] Latest frame preview: {os.path.join(args.output_dir, 'preview_latest.png')}")
