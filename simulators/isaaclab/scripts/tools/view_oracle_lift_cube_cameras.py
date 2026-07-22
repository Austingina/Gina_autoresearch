# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Visualize end-effector and third-person camera feeds while running the lift-cube oracle.

By default, previews are written to disk (no OpenCV GUI required). This works on
headless OpenCV builds that lack GTK/Cocoa support.

Example:

.. code-block:: bash

    ./isaaclab.sh -p scripts/tools/view_oracle_lift_cube_cameras.py --num_envs 1

    # In another terminal, watch the live preview image:
    watch -n 0.2 xdg-open runs/lift_cube_camera_preview/preview_latest.png

    # Or play the generated mp4 after the run:
    ffplay runs/lift_cube_camera_preview/preview.mp4

Optional OpenCV window (requires opencv-python with GUI support):

.. code-block:: bash

    ./isaaclab.sh -p scripts/tools/view_oracle_lift_cube_cameras.py --show_opencv
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize lift-cube oracle camera feeds during data collection.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Lift-Cube-Franka-Oracle-Data-v0",
    help="Task with ee_camera and third_person_camera configured.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. Use 1 for visualization.")
parser.add_argument(
    "--position_threshold",
    type=float,
    default=0.01,
    help="Position threshold for the pick-and-lift state machine.",
)
parser.add_argument(
    "--max_episode_steps",
    type=int,
    default=400,
    help="Reset the episode after this many steps.",
)
parser.add_argument(
    "--max_preview_steps",
    type=int,
    default=0,
    help="Stop after this many simulation steps. 0 means run until Isaac Sim is closed.",
)
parser.add_argument(
    "--display_scale",
    type=int,
    default=4,
    help="Upscale factor for saved preview images and video.",
)
parser.add_argument("--camera_height", type=int, default=84, help="RGB camera height in pixels.")
parser.add_argument("--camera_width", type=int, default=84, help="RGB camera width in pixels.")
parser.add_argument(
    "--output_dir",
    type=str,
    default="./runs/lift_cube_camera_preview",
    help="Directory for preview_latest.png, frame dumps, and preview.mp4.",
)
parser.add_argument(
    "--save_frames",
    action="store_true",
    default=False,
    help="Also save every frame as frame_XXXXXX.png under output_dir/frames/.",
)
parser.add_argument(
    "--video_fps",
    type=int,
    default=20,
    help="FPS for preview.mp4 written under output_dir.",
)
parser.add_argument(
    "--show_opencv",
    action="store_true",
    default=False,
    help="Try to open an OpenCV window. Requires opencv-python with GUI support.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.mdp.oracle import LiftCubeOraclePolicy
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class PreviewWriter:
    """Write camera previews to disk and optionally to an OpenCV window."""

    def __init__(self, output_dir: str, video_fps: int, save_frames: bool, show_opencv: bool):
        self.output_dir = output_dir
        self.frames_dir = os.path.join(output_dir, "frames")
        self.latest_path = os.path.join(output_dir, "preview_latest.png")
        self.video_path = os.path.join(output_dir, "preview.mp4")
        self.save_frames = save_frames
        self.show_opencv = show_opencv
        self.frame_index = 0
        self._video_writer = None
        self._opencv_enabled = False
        self._window_name = "lift_cube_oracle_cameras"

        os.makedirs(output_dir, exist_ok=True)
        if save_frames:
            os.makedirs(self.frames_dir, exist_ok=True)

        if show_opencv:
            try:
                cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
                self._opencv_enabled = True
                print("[INFO] OpenCV window enabled.")
            except cv2.error as exc:
                print(f"[WARN] OpenCV GUI unavailable: {exc}")
                print("[WARN] Continuing with file-based preview only.")

    def write(self, panel_bgr: np.ndarray) -> int | None:
        """Save panel image and optionally display it. Returns a key code if OpenCV GUI is active."""
        cv2.imwrite(self.latest_path, panel_bgr)

        if self.save_frames:
            frame_path = os.path.join(self.frames_dir, f"frame_{self.frame_index:06d}.png")
            cv2.imwrite(frame_path, panel_bgr)

        if self._video_writer is None:
            height, width = panel_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(self.video_path, fourcc, args_cli.video_fps, (width, height))
            if not self._video_writer.isOpened():
                print(f"[WARN] Failed to open video writer at {self.video_path}")
                self._video_writer = None
        if self._video_writer is not None:
            self._video_writer.write(panel_bgr)

        self.frame_index += 1

        if not self._opencv_enabled:
            return None
        cv2.imshow(self._window_name, panel_bgr)
        return cv2.waitKey(1) & 0xFF

    def close(self):
        if self._video_writer is not None:
            self._video_writer.release()
        if self._opencv_enabled:
            cv2.destroyAllWindows()


def _to_display_image(rgb_tensor: torch.Tensor, scale: int) -> np.ndarray:
    """Convert a single-env RGB tensor to a BGR image for OpenCV."""
    rgb = rgb_tensor.detach().cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if scale > 1:
        bgr = cv2.resize(bgr, (bgr.shape[1] * scale, bgr.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    return bgr


def _read_camera_rgb(env, camera_name: str, env_index: int = 0) -> torch.Tensor:
    """Read RGB output from a scene camera sensor."""
    camera = env.scene[camera_name]
    return camera.data.output["rgb"][env_index]


def _make_panel(ee_bgr: np.ndarray, third_person_bgr: np.ndarray, step: int, reached_goal: bool) -> np.ndarray:
    """Stack the two camera views side by side with labels."""
    ee_view = ee_bgr.copy()
    third_view = third_person_bgr.copy()
    label_color = (0, 255, 0) if reached_goal else (255, 255, 255)
    cv2.putText(ee_view, "ee_camera", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1, cv2.LINE_AA)
    cv2.putText(
        third_view,
        "third_person_camera",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        label_color,
        1,
        cv2.LINE_AA,
    )
    panel = np.concatenate([ee_view, third_view], axis=1)
    cv2.putText(
        panel,
        f"step={step}",
        (8, panel.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return panel


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None

    if hasattr(env_cfg.scene, "ee_camera"):
        env_cfg.scene.ee_camera.height = args_cli.camera_height
        env_cfg.scene.ee_camera.width = args_cli.camera_width
    if hasattr(env_cfg.scene, "third_person_camera"):
        env_cfg.scene.third_person_camera.height = args_cli.camera_height
        env_cfg.scene.third_person_camera.width = args_cli.camera_width

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    oracle = LiftCubeOraclePolicy(env, position_threshold=args_cli.position_threshold)
    preview = PreviewWriter(
        output_dir=os.path.abspath(args_cli.output_dir),
        video_fps=args_cli.video_fps,
        save_frames=args_cli.save_frames,
        show_opencv=args_cli.show_opencv,
    )

    print(f"[INFO] Visualizing task: {args_cli.task}")
    print(f"[INFO] Camera resolution: {args_cli.camera_width}x{args_cli.camera_height}")
    print(f"[INFO] Live preview image: {preview.latest_path}")
    print(f"[INFO] Preview video: {preview.video_path}")
    if preview._opencv_enabled:
        print("[INFO] OpenCV controls: r=reset, q/Esc=quit")
    else:
        print("[INFO] Close Isaac Sim to stop. Preview image updates every step.")

    env.reset()
    oracle.reset()
    episode_step = 0
    total_step = 0
    manual_reset = False

    while simulation_app.is_running():
        with torch.inference_mode():
            if manual_reset:
                env.reset()
                oracle.reset()
                episode_step = 0
                manual_reset = False

            actions = oracle.compute()
            _, _, terminated, truncated, _ = env.step(actions)
            episode_step += 1
            total_step += 1

            if terminated.any() or truncated.any() or episode_step >= args_cli.max_episode_steps:
                done_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
                if done_ids.numel() > 0:
                    oracle.reset(done_ids.tolist() if done_ids.ndim > 0 else [int(done_ids.item())])
                env.reset()
                oracle.reset()
                episode_step = 0

            ee_rgb = _read_camera_rgb(env, "ee_camera", env_index=0)
            third_rgb = _read_camera_rgb(env, "third_person_camera", env_index=0)
            ee_bgr = _to_display_image(ee_rgb, args_cli.display_scale)
            third_bgr = _to_display_image(third_rgb, args_cli.display_scale)
            reached_goal = bool(mdp.object_reached_goal(env, threshold=0.02)[0].item())
            panel = _make_panel(ee_bgr, third_bgr, episode_step, reached_goal)
            key = preview.write(panel)

            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                manual_reset = True

            if args_cli.max_preview_steps > 0 and total_step >= args_cli.max_preview_steps:
                print(f"[INFO] Reached max_preview_steps={args_cli.max_preview_steps}. Stopping.")
                break

            if env.sim.is_stopped():
                break

    preview.close()
    env.close()
    print(f"[INFO] Saved latest preview to: {preview.latest_path}")
    print(f"[INFO] Saved preview video to: {preview.video_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
