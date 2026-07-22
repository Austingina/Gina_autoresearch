# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Live preview helpers for oracle data-collection scripts."""

from __future__ import annotations

import os

import cv2
import numpy as np

from isaaclab_tasks.manager_based.manipulation.lift.mdp.data_collection import OracleStepData, compute_oracle_step_data


class CollectionPreviewWriter:
    """Write live collection previews to disk and optionally to an OpenCV window."""

    def __init__(
        self,
        output_dir: str,
        *,
        video_fps: int = 20,
        show_opencv: bool = False,
        window_name: str = "oracle_collection_preview",
    ):
        self.output_dir = os.path.abspath(output_dir)
        self.latest_path = os.path.join(self.output_dir, "preview_latest.png")
        self.video_path = os.path.join(self.output_dir, "preview.mp4")
        self.window_name = window_name
        self._video_writer = None
        self._opencv_enabled = False

        os.makedirs(self.output_dir, exist_ok=True)

        if show_opencv:
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                self._opencv_enabled = True
                print("[INFO] OpenCV preview window enabled.")
            except cv2.error as exc:
                print(f"[WARN] OpenCV GUI unavailable: {exc}")
                print("[WARN] Continuing with file-based preview only.")

        self._video_fps = video_fps

    def write(self, panel_bgr: np.ndarray) -> int | None:
        """Save the panel and optionally display it."""
        cv2.imwrite(self.latest_path, panel_bgr)

        if self._video_writer is None:
            height, width = panel_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(self.video_path, fourcc, self._video_fps, (width, height))
            if not self._video_writer.isOpened():
                print(f"[WARN] Failed to open preview video writer at {self.video_path}")
                self._video_writer = None
        if self._video_writer is not None:
            self._video_writer.write(panel_bgr)

        if not self._opencv_enabled:
            return None
        cv2.imshow(self.window_name, panel_bgr)
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        if self._video_writer is not None:
            self._video_writer.release()
        if self._opencv_enabled:
            cv2.destroyAllWindows()


def read_camera_bgr(env, camera_name: str, env_index: int, display_scale: int) -> np.ndarray:
    """Read one environment's RGB camera output as a BGR image."""
    rgb = env.scene[camera_name].data.output["rgb"][env_index].detach().cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if display_scale > 1:
        bgr = cv2.resize(
            bgr,
            (bgr.shape[1] * display_scale, bgr.shape[0] * display_scale),
            interpolation=cv2.INTER_NEAREST,
        )
    return bgr


def _world_xy_to_pixel(x: float, y: float, plot_size: int, xlim: tuple[float, float], ylim: tuple[float, float]) -> tuple[int, int]:
    px = int(round((x - xlim[0]) / (xlim[1] - xlim[0]) * (plot_size - 1)))
    py = int(round((1.0 - (y - ylim[0]) / (ylim[1] - ylim[0])) * (plot_size - 1)))
    return px, py


def _draw_xy_arrow(
    canvas: np.ndarray,
    origin_xy: tuple[float, float],
    direction_xy: np.ndarray,
    *,
    plot_size: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    length_m: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    direction = np.asarray(direction_xy, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return
    direction = direction / norm
    end_xy = (origin_xy[0] + direction[0] * length_m, origin_xy[1] + direction[1] * length_m)
    start_px = _world_xy_to_pixel(origin_xy[0], origin_xy[1], plot_size, xlim, ylim)
    end_px = _world_xy_to_pixel(end_xy[0], end_xy[1], plot_size, xlim, ylim)
    cv2.arrowedLine(canvas, start_px, end_px, color, thickness, cv2.LINE_AA, tipLength=0.25)


def render_topdown_bgr(
    ee_pos: np.ndarray,
    goal_pos: np.ndarray,
    lin_vel: np.ndarray,
    *,
    goal_phase: float,
    gripper_rate: float | None,
    plot_size: int,
    velocity_scale: float,
    cube_pos: np.ndarray | None = None,
    dist_to_goal: float | None = None,
    goal_success_radius_m: float = 0.02,
    gt_subgoal_xy: np.ndarray | None = None,
    xlim: tuple[float, float] = (0.2, 0.8),
    ylim: tuple[float, float] = (-0.35, 0.35),
) -> np.ndarray:
    """Render top-down EE / cube / goal / velocity view with OpenCV."""
    canvas = np.full((plot_size, plot_size, 3), 255, dtype=np.uint8)
    m_per_px = (xlim[1] - xlim[0]) / max(plot_size - 1, 1)

    for x_grid in np.linspace(xlim[0], xlim[1], 7):
        p0 = _world_xy_to_pixel(x_grid, ylim[0], plot_size, xlim, ylim)
        p1 = _world_xy_to_pixel(x_grid, ylim[1], plot_size, xlim, ylim)
        cv2.line(canvas, p0, p1, (230, 230, 230), 1, cv2.LINE_AA)
    for y_grid in np.linspace(ylim[0], ylim[1], 7):
        p0 = _world_xy_to_pixel(xlim[0], y_grid, plot_size, xlim, ylim)
        p1 = _world_xy_to_pixel(xlim[1], y_grid, plot_size, xlim, ylim)
        cv2.line(canvas, p0, p1, (230, 230, 230), 1, cv2.LINE_AA)

    ee_px = _world_xy_to_pixel(float(ee_pos[0]), float(ee_pos[1]), plot_size, xlim, ylim)
    goal_px = _world_xy_to_pixel(float(goal_pos[0]), float(goal_pos[1]), plot_size, xlim, ylim)

    # Success threshold disk around goal (xy plane).
    success_r_px = max(2, int(round(goal_success_radius_m / m_per_px)))
    cv2.circle(canvas, goal_px, success_r_px, (0, 200, 0), 1, cv2.LINE_AA)
    cv2.circle(canvas, goal_px, success_r_px, (220, 255, 220), -1, cv2.LINE_AA)

    if cube_pos is not None:
        cube_px = _world_xy_to_pixel(float(cube_pos[0]), float(cube_pos[1]), plot_size, xlim, ylim)
        cv2.line(canvas, cube_px, goal_px, (180, 80, 220), 2, cv2.LINE_AA)
        cv2.circle(canvas, cube_px, 7, (0, 140, 0), -1, cv2.LINE_AA)
        if dist_to_goal is None:
            dist_to_goal = float(
                np.linalg.norm(
                    np.array([cube_pos[0] - goal_pos[0], cube_pos[1] - goal_pos[1], cube_pos[2] - goal_pos[2]])
                )
            )
        mid = ((cube_px[0] + goal_px[0]) // 2, (cube_px[1] + goal_px[1]) // 2)
        cv2.putText(
            canvas,
            f"dist={dist_to_goal * 100:.1f}cm",
            (mid[0] - 40, mid[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (120, 40, 160),
            1,
            cv2.LINE_AA,
        )

    # Goal marker: red star + label.
    cv2.drawMarker(canvas, goal_px, (0, 0, 220), markerType=cv2.MARKER_STAR, markerSize=14, thickness=2)
    cv2.putText(
        canvas,
        "GOAL",
        (goal_px[0] + 10, goal_px[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.circle(canvas, ee_px, 6, (255, 128, 0), -1, cv2.LINE_AA)

    speed = float(np.linalg.norm(lin_vel[:2]))
    if speed > 1e-6:
        direction = lin_vel[:2] / speed
        arrow_len = speed * velocity_scale
        end_xy = (
            float(ee_pos[0] + direction[0] * arrow_len),
            float(ee_pos[1] + direction[1] * arrow_len),
        )
        end_px = _world_xy_to_pixel(end_xy[0], end_xy[1], plot_size, xlim, ylim)
        cv2.arrowedLine(canvas, ee_px, end_px, (255, 200, 0), 2, cv2.LINE_AA, tipLength=0.25)

    if gt_subgoal_xy is not None:
        _draw_xy_arrow(
            canvas,
            (float(ee_pos[0]), float(ee_pos[1])),
            gt_subgoal_xy,
            plot_size=plot_size,
            xlim=xlim,
            ylim=ylim,
            length_m=0.10,
            color=(0, 180, 0),
            thickness=2,
        )

    phase_text = "lift" if goal_phase >= 0.5 else "grasp"
    info = f"phase={phase_text} |v|={speed:.3f}"
    if gripper_rate is not None:
        info += f" grip={gripper_rate:+.3f}"
    if dist_to_goal is not None:
        info += f" dist_goal={dist_to_goal * 100:.1f}cm"
    cv2.putText(canvas, info, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    legend = "orange=policy xy  green=GT subgoal xy  ★goal  ●cube"
    cv2.putText(canvas, legend, (8, plot_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1, cv2.LINE_AA)
    return canvas


def render_xz_direction_bgr(
    ee_pos: np.ndarray,
    lin_vel: np.ndarray,
    *,
    guidance_xz: np.ndarray | None = None,
    gt_subgoal_xz: np.ndarray | None = None,
    plot_size: int,
    arrow_scale: float = 0.12,
    xlim: tuple[float, float] = (0.2, 0.8),
    zlim: tuple[float, float] = (-0.05, 0.55),
) -> np.ndarray:
    """Side x-z panel: compare policy velocity vs Gθ vs GT subgoal (training plane)."""
    canvas = np.full((plot_size, plot_size, 3), 255, dtype=np.uint8)

    def _xz_to_pixel(x: float, z: float) -> tuple[int, int]:
        px = int(round((x - xlim[0]) / (xlim[1] - xlim[0]) * (plot_size - 1)))
        pz = int(round((1.0 - (z - zlim[0]) / (zlim[1] - zlim[0])) * (plot_size - 1)))
        return px, pz

    for x_grid in np.linspace(xlim[0], xlim[1], 7):
        p0 = _xz_to_pixel(x_grid, zlim[0])
        p1 = _xz_to_pixel(x_grid, zlim[1])
        cv2.line(canvas, p0, p1, (230, 230, 230), 1, cv2.LINE_AA)
    for z_grid in np.linspace(zlim[0], zlim[1], 7):
        p0 = _xz_to_pixel(xlim[0], z_grid)
        p1 = _xz_to_pixel(xlim[1], z_grid)
        cv2.line(canvas, p0, p1, (230, 230, 230), 1, cv2.LINE_AA)

    ee_x, ee_z = float(ee_pos[0]), float(ee_pos[2])
    ee_px = _xz_to_pixel(ee_x, ee_z)
    cv2.circle(canvas, ee_px, 6, (255, 128, 0), -1, cv2.LINE_AA)

    def _draw_xz_arrow(direction_xz: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> None:
        direction = np.asarray(direction_xz, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return
        direction = direction / norm
        end_x = ee_x + direction[0] * arrow_scale
        end_z = ee_z + direction[1] * arrow_scale
        cv2.arrowedLine(canvas, ee_px, _xz_to_pixel(end_x, end_z), color, thickness, cv2.LINE_AA, tipLength=0.25)

    policy_xz = np.array([lin_vel[0], lin_vel[2]], dtype=np.float64)
    _draw_xz_arrow(policy_xz, (255, 200, 0))
    if gt_subgoal_xz is not None:
        _draw_xz_arrow(gt_subgoal_xz, (0, 180, 0))
    if guidance_xz is not None:
        _draw_xz_arrow(guidance_xz, (200, 0, 200))

    cos_gt = ""
    if guidance_xz is not None and gt_subgoal_xz is not None:
        g = np.asarray(guidance_xz, dtype=np.float64)
        t = np.asarray(gt_subgoal_xz, dtype=np.float64)
        gn, tn = np.linalg.norm(g), np.linalg.norm(t)
        if gn > 1e-6 and tn > 1e-6:
            cos_gt = f" cos(Gθ,G*)={float(np.dot(g / gn, t / tn)):.2f}"

    cv2.putText(canvas, "x-z directions from EE", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"orange=policy purple=Gθ green=G*{cos_gt}",
        (8, plot_size - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )
    return canvas


def build_collection_preview_panel(
    env,
    env_index: int,
    step_data: OracleStepData,
    *,
    episode_step: int,
    exported_demo_count: int,
    display_scale: int,
    velocity_scale: float,
    using_noise: bool = False,
    noise_flag: int = 0,
    reached_goal: bool = False,
    camera_name: str = "third_person_camera",
) -> np.ndarray:
    """Build camera + top-down preview panel for one environment."""
    camera_bgr = read_camera_bgr(env, camera_name, env_index, display_scale)
    plot_size = camera_bgr.shape[0]

    ee_pos = step_data.ee_pose[env_index, :3].detach().cpu().numpy()
    goal_pos = step_data.goal_position[env_index].detach().cpu().numpy()
    lin_vel = step_data.ee_velocity[env_index, :3].detach().cpu().numpy()
    goal_phase = float(step_data.goal_phase[env_index, 0].item())
    gripper_rate = float(step_data.ee_velocity[env_index, 6].item()) if step_data.ee_velocity.shape[-1] >= 7 else None

    topdown_bgr = render_topdown_bgr(
        ee_pos,
        goal_pos,
        lin_vel,
        goal_phase=goal_phase,
        gripper_rate=gripper_rate,
        plot_size=plot_size,
        velocity_scale=velocity_scale,
    )
    panel = np.concatenate([camera_bgr, topdown_bgr], axis=1)

    status_color = (0, 220, 0) if reached_goal else (255, 255, 255)
    cv2.putText(
        panel,
        f"env={env_index} step={episode_step} demos={exported_demo_count}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        status_color,
        2,
        cv2.LINE_AA,
    )
    if using_noise:
        noise_color = (0, 140, 255) if noise_flag > 0 else (180, 180, 180)
        cv2.putText(
            panel,
            f"noise_flag={noise_flag}",
            (8, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            noise_color,
            1,
            cv2.LINE_AA,
        )
    return panel


def get_step_data_for_preview(env, control_dt: float) -> OracleStepData:
    """Compute the same step signals used by dataset recorders."""
    return compute_oracle_step_data(env, control_dt=control_dt)


def maybe_create_preview_writer(args) -> CollectionPreviewWriter | None:
    """Create a preview writer if enabled in CLI args."""
    if not getattr(args, "enable_preview", False):
        return None
    writer = CollectionPreviewWriter(
        output_dir=args.preview_dir,
        video_fps=args.preview_fps,
        show_opencv=args.show_opencv,
    )
    print(f"[INFO] Live preview image: {writer.latest_path}")
    print(f"[INFO] Live preview video: {writer.video_path}")
    return writer
