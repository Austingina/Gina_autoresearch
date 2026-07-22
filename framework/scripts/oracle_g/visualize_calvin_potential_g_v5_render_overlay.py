#!/usr/bin/env python3
"""Overlay CALVIN G-v5 potential-field tokens on rendered camera images."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

for _alias, _target in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
}.items():
    if _alias not in np.__dict__:
        setattr(np, _alias, _target)

SCRIPT_DIR = Path(__file__).resolve().parent
POLICY_SCRIPT_DIR = SCRIPT_DIR.parents[0] / "policy_state_bc"
sys.path.insert(0, POLICY_SCRIPT_DIR.as_posix())

import train_eval_state_bc_task_policy as pol  # noqa: E402


TOKEN_COUNT = pol.ARTICULATION_V5_TOKEN_COUNT
TOKEN_DIM = pol.ARTICULATION_V5_TOKEN_DIM

G_MODE_TO_FN = {
    "v5_potential": pol.make_articulation_potential_g_v5,
    "v5_attraction_only": pol.make_articulation_potential_g_v5_attraction_only,
    "v6_interaction_field": pol.make_articulation_interaction_field_g_v6,
    "v7_unified_field": pol.make_articulation_unified_field_g_v7,
}


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def unwrap_env(env: Any) -> Any:
    cur = env
    for _ in range(10):
        if hasattr(cur, "cameras"):
            return cur
        nxt = getattr(cur, "env", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return env


def find_camera(env: Any, camera_name: str) -> Any:
    base_env = unwrap_env(env)
    for camera in getattr(base_env, "cameras", []) or []:
        if getattr(camera, "name", None) == camera_name:
            return camera
    names = [getattr(camera, "name", "<unnamed>") for camera in getattr(base_env, "cameras", []) or []]
    raise RuntimeError(f"Camera {camera_name!r} not found. Available cameras: {names}")


def project_point(camera: Any, point_xyz: np.ndarray) -> tuple[int, int] | None:
    point = np.asarray([point_xyz[0], point_xyz[1], point_xyz[2], 1.0], dtype=np.float64)
    persp_m = np.asarray(camera.projectionMatrix, dtype=np.float64).reshape((4, 4)).T
    view_m = np.asarray(camera.viewMatrix, dtype=np.float64).reshape((4, 4)).T
    clip = persp_m @ view_m @ point
    if abs(float(clip[-1])) < 1e-8:
        return None
    ndc = clip[:3] / clip[-1]
    if not np.all(np.isfinite(ndc)):
        return None
    x = int(np.floor((ndc[0] + 1.0) * 0.5 * int(camera.width)))
    y = int(np.floor((1.0 - (ndc[1] + 1.0) * 0.5) * int(camera.height)))
    margin = 24
    if x < -margin or x >= int(camera.width) + margin or y < -margin or y >= int(camera.height) + margin:
        return None
    return x, y


def draw_text(img: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.45) -> None:
    x, y = xy
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_arrow(img: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.arrowedLine(img, p0, p1, color, 2, cv2.LINE_AA, tipLength=0.28)
    cv2.circle(img, p0, 4, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, p0, 3, color, -1, cv2.LINE_AA)


def decode_tokens(robot: np.ndarray, tokens_flat: np.ndarray) -> list[dict[str, Any]]:
    tokens = tokens_flat.reshape(TOKEN_COUNT, TOKEN_DIM)
    decoded: list[dict[str, Any]] = []
    for i, token in enumerate(tokens):
        anchor_delta = token[:3]
        decoded.append(
            {
                "index": i,
                "anchor": robot[:3] + anchor_delta,
                "anchor_delta": anchor_delta,
                "attractive_flow": token[3:6],
                "repulsive_flow": token[6:9],
                "attract_weight": float(token[9]),
                "repel_weight": float(token[10]),
            }
        )
    return decoded


def joint_points(joint_model: dict[str, Any], q: float) -> dict[str, np.ndarray]:
    intercept = np.asarray(joint_model["intercept"], dtype=np.float32)
    slope = np.asarray(joint_model["slope"], dtype=np.float32)
    q_min = float(joint_model.get("q_min_sample", 0.0))
    q_max = float(joint_model.get("q_max_sample", q_min + float(joint_model["joint_range"])))
    return {
        "current": intercept + slope * np.float32(q),
        "min": intercept + slope * np.float32(q_min),
        "max": intercept + slope * np.float32(q_max),
    }


def draw_projected_line(
    img: np.ndarray,
    camera: Any,
    p0_world: np.ndarray,
    p1_world: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> bool:
    p0 = project_point(camera, p0_world)
    p1 = project_point(camera, p1_world)
    if p0 is None or p1 is None:
        return False
    cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)
    return True


def overlay_g_on_image(
    rgb: np.ndarray,
    camera: Any,
    robot: np.ndarray,
    scene: np.ndarray,
    task: str,
    geometry: dict[str, Any],
    g_mode: str,
    flow_world_scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    img = np.asarray(rgb).copy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    g_fn = G_MODE_TO_FN[g_mode]
    tokens_flat = g_fn(
        robot[None, :],
        scene[None, :],
        np.asarray([task], dtype=object),
        geometry,
    )[0]
    decoded = decode_tokens(robot, tokens_flat)

    joint_name = pol.ARTICULATION_TASK_TO_JOINT[task]
    joint_model = geometry["joints"][joint_name]
    joint_idx = int(joint_model["joint_index"])
    q = float(scene[joint_idx])
    target = float(pol.articulation_target_controls(task, scene)[joint_idx])
    residual_norm = (target - q) / max(float(joint_model["joint_range"]), 1e-6)
    pts = joint_points(joint_model, q)

    projected: list[dict[str, Any]] = []
    draw_projected_line(img, camera, pts["min"], pts["max"], (120, 120, 120), 2)

    current_px = project_point(camera, pts["current"])
    if current_px is not None:
        cv2.drawMarker(img, current_px, (255, 165, 0), markerType=cv2.MARKER_STAR, markerSize=17, thickness=2)

    eef_px = project_point(camera, robot[:3])
    if eef_px is not None:
        cv2.drawMarker(img, eef_px, (255, 255, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=16, thickness=3)
        cv2.drawMarker(img, eef_px, (0, 0, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=2)

    for token in decoded:
        anchor = token["anchor"]
        anchor_px = project_point(camera, anchor)
        token_record = dict(token)
        token_record["anchor_px"] = anchor_px
        if token["attract_weight"] == 0.0 and token["repel_weight"] == 0.0:
            token_record["visible"] = False
            projected.append(token_record)
            continue
        token_record["visible"] = True
        if anchor_px is None:
            projected.append(token_record)
            continue

        cv2.circle(img, anchor_px, 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, anchor_px, 4, (210, 210, 210), -1, cv2.LINE_AA)
        draw_text(img, str(token["index"]), (anchor_px[0] + 5, anchor_px[1] - 5), scale=0.38)

        att = np.asarray(token["attractive_flow"], dtype=np.float32)
        rep = np.asarray(token["repulsive_flow"], dtype=np.float32)
        if float(np.linalg.norm(att)) > 1e-5:
            end_px = project_point(camera, anchor + att * flow_world_scale)
            token_record["attraction_end_px"] = end_px
            if end_px is not None:
                draw_arrow(img, anchor_px, end_px, (0, 210, 0))
        if float(np.linalg.norm(rep)) > 1e-5:
            end_px = project_point(camera, anchor + rep * flow_world_scale)
            token_record["repulsion_end_px"] = end_px
            if end_px is not None:
                draw_arrow(img, anchor_px, end_px, (220, 40, 40))
        projected.append(token_record)

    legend = "green unified field | star part(q) | x eef" if g_mode == "v7_unified_field" else "green attraction | red repulsion | star part(q) | x eef"
    label_lines = [
        f"{g_mode} | {task} | q={q:.3f} target={target:.3f} res={residual_norm:+.2f}",
        legend,
    ]
    y = 20
    for line in label_lines:
        draw_text(img, line, (8, y), scale=0.45)
        y += 20

    meta = {
        "task": task,
        "g_mode": g_mode,
        "camera": getattr(camera, "name", None),
        "q": q,
        "target": target,
        "residual_norm": residual_norm,
        "robot_eef": robot[:3],
        "part_current": pts["current"],
        "part_current_px": current_px,
        "eef_px": eef_px,
        "tokens": projected,
    }
    return img, meta


def make_grid(images: list[np.ndarray]) -> np.ndarray:
    if not images:
        raise RuntimeError("No images to compose")
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    h = max(heights)
    padded = []
    for img in images:
        if img.shape[0] == h:
            padded.append(img)
            continue
        pad = np.zeros((h - img.shape[0], img.shape[1], 3), dtype=img.dtype)
        padded.append(np.concatenate([img, pad], axis=0))
    return np.concatenate(padded, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calvin-code", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--geometry-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["open_drawer", "close_drawer", "move_slider_left", "move_slider_right"],
    )
    parser.add_argument("--camera", default="static")
    parser.add_argument("--g-mode", choices=sorted(G_MODE_TO_FN), default="v5_potential")
    parser.add_argument("--samples-per-task", type=int, default=3)
    parser.add_argument("--scan-sequences", type=int, default=5000)
    parser.add_argument("--flow-world-scale", type=float, default=0.12)
    args = parser.parse_args()

    args.calvin_code = args.calvin_code.resolve()
    args.data_root = args.data_root.resolve()
    args.geometry_json = args.geometry_json.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    calvin_models = args.calvin_code / "calvin_models"
    calvin_env = args.calvin_code / "calvin_env"
    calvin_agent = calvin_models / "calvin_agent"
    sys.path.insert(0, calvin_models.as_posix())
    sys.path.insert(0, calvin_env.as_posix())
    os.chdir(calvin_agent)

    from calvin_agent.evaluation import evaluate_policy as eval_mod
    from calvin_agent.evaluation.multistep_sequences import get_sequences

    with open(args.geometry_json, "r", encoding="utf-8") as f:
        geometry = json.load(f)

    buckets: dict[str, list[Any]] = {task: [] for task in args.tasks}
    for initial_state, seq in get_sequences(args.scan_sequences):
        first_task = seq[0]
        if first_task in buckets and len(buckets[first_task]) < args.samples_per_task:
            buckets[first_task].append(initial_state)
        if all(len(v) >= args.samples_per_task for v in buckets.values()):
            break

    missing = {task: len(states) for task, states in buckets.items() if len(states) < args.samples_per_task}
    if missing:
        raise RuntimeError(f"Could not find enough validation first-task states: {missing}")

    env = eval_mod.make_env(args.data_root)
    camera = find_camera(env, args.camera)
    summary: dict[str, Any] = {
        "data_root": args.data_root,
        "geometry_json": args.geometry_json,
        "camera": args.camera,
        "g_mode": args.g_mode,
        "flow_world_scale": args.flow_world_scale,
        "tasks": {},
    }

    for task, initial_states in buckets.items():
        images: list[np.ndarray] = []
        records: list[dict[str, Any]] = []
        for sample_idx, initial_state in enumerate(initial_states):
            robot_obs, scene_obs = eval_mod.get_env_state_for_initial_condition(initial_state)
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            obs = env.get_obs()
            rgb_key = f"rgb_{args.camera}"
            if rgb_key not in obs["rgb_obs"]:
                raise RuntimeError(f"Missing {rgb_key}; available: {sorted(obs['rgb_obs'].keys())}")
            img, meta = overlay_g_on_image(
                obs["rgb_obs"][rgb_key],
                camera,
                np.asarray(obs["robot_obs"], dtype=np.float32),
                np.asarray(obs["scene_obs"], dtype=np.float32),
                task,
                geometry,
                g_mode=args.g_mode,
                flow_world_scale=args.flow_world_scale,
            )
            draw_text(img, f"sample {sample_idx}", (8, img.shape[0] - 12), scale=0.45)
            image_path = args.out_dir / f"g_render_overlay_{args.g_mode}_{task}_sample{sample_idx:02d}_{args.camera}.png"
            cv2.imwrite(image_path.as_posix(), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            meta["sample_idx"] = sample_idx
            meta["initial_state"] = initial_state
            meta["image"] = image_path
            records.append(meta)
            images.append(img)

        grid = make_grid(images)
        grid_path = args.out_dir / f"g_render_overlay_{args.g_mode}_{task}_{args.camera}.png"
        cv2.imwrite(grid_path.as_posix(), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        summary["tasks"][task] = {"grid": grid_path, "samples": records}
        print(f"[overlay] {task}: {grid_path}", flush=True)

    summary_path = args.out_dir / "g_v5_render_overlay_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=json_default)
    print(f"[summary] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
