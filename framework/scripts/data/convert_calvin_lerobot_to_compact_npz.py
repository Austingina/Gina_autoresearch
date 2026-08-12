#!/usr/bin/env python3
"""Convert CALVIN LeRobot episodes to compact NPZ records with trajectory G."""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import yaml
from PIL import Image


PUSH_TASKS = {
    f"push_{color}_block_{direction}": (
        color,
        np.asarray([sign, 0.0, 0.0], dtype=np.float32),
    )
    for color in ("red", "blue", "pink")
    for direction, sign in (("left", -1.0), ("right", 1.0))
}


def color_mask(image: np.ndarray, color: str) -> np.ndarray:
    rgb = image.astype(np.int16)
    red, green, blue = (rgb[..., index] for index in range(3))
    if color == "red":
        mask = (red > 140) & (green < 130) & (blue < 130) & (red - green > 60) & (red - blue > 60)
    elif color == "blue":
        mask = (blue > 120) & (red < 150) & (green < 170) & (blue - red > 45) & (blue - green > 20)
    elif color == "pink":
        mask = (red > 130) & (blue > 130) & (green < 150) & (red - green > 50) & (blue - green > 50)
    else:
        raise ValueError(color)
    return mask.astype(np.uint8)


def component_centers(image: np.ndarray, color: str) -> list[np.ndarray]:
    mask = color_mask(image, color)
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [
        np.asarray(centroids[label], dtype=np.float32)
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= 8
    ]


def camera_matrix(config: dict[str, object]) -> np.ndarray:
    eye = np.asarray(config["look_from"], dtype=np.float64)
    target = np.asarray(config["look_at"], dtype=np.float64)
    up = np.asarray(config["up_vector"], dtype=np.float64)
    forward = (target - eye) / np.linalg.norm(target - eye)
    side = np.cross(forward, up)
    side /= np.linalg.norm(side)
    camera_up = np.cross(side, forward)
    view = np.eye(4, dtype=np.float64)
    view[:3, :3] = np.stack([side, camera_up, -forward], axis=0)
    view[:3, 3] = -view[:3, :3] @ eye

    fov = np.deg2rad(float(config["fov"]))
    aspect = float(config["aspect"])
    near = float(config["nearval"])
    far = float(config["farval"])
    focal = 1.0 / np.tan(fov / 2.0)
    projection = np.zeros((4, 4), dtype=np.float64)
    projection[0, 0] = focal / aspect
    projection[1, 1] = focal
    projection[2, 2] = (far + near) / (near - far)
    projection[2, 3] = 2.0 * far * near / (near - far)
    projection[3, 2] = -1.0
    return projection @ view


def project(points: np.ndarray, matrix: np.ndarray, width: int, height: int) -> np.ndarray:
    homogeneous = np.concatenate(
        [points.astype(np.float64), np.ones((len(points), 1), dtype=np.float64)], axis=1
    )
    clip = (matrix @ homogeneous.T).T
    normalized = clip[:, :3] / clip[:, 3:4]
    return np.stack(
        [(normalized[:, 0] + 1.0) * width / 2.0, (1.0 - normalized[:, 1]) * height / 2.0],
        axis=1,
    ).astype(np.float32)


def decode_images(column: list[dict[str, object]]) -> list[np.ndarray]:
    return [np.asarray(Image.open(io.BytesIO(item["bytes"])).convert("RGB")) for item in column]


def infer_contact(
    images: list[np.ndarray],
    eef: np.ndarray,
    color: str,
    matrix: np.ndarray,
    width: int,
    height: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    eef_pixels = project(eef, matrix, width, height)
    centers = np.full((len(images), 2), np.nan, dtype=np.float32)
    pixel_distances = np.full(len(images), np.inf, dtype=np.float32)
    for index, image in enumerate(images):
        candidates = component_centers(image, color)
        if not candidates:
            continue
        candidate_array = np.stack(candidates)
        selected = int(np.argmin(np.linalg.norm(candidate_array - eef_pixels[index], axis=1)))
        centers[index] = candidate_array[selected]
        pixel_distances[index] = float(np.linalg.norm(centers[index] - eef_pixels[index]))

    valid = np.flatnonzero(np.isfinite(centers[:, 0]))
    if len(valid) == 0:
        contact_index = int(np.argmin(np.linalg.norm(eef - np.median(eef, axis=0), axis=1)))
        return contact_index, centers, pixel_distances
    for index in range(len(images)):
        left = max(index - 2, 0)
        right = min(index + 3, len(images) - 1)
        if not np.isfinite(centers[[left, right]]).all():
            continue
        motion = float(np.linalg.norm(centers[right] - centers[left]))
        if pixel_distances[index] <= 25.0 and motion >= 1.5:
            return index, centers, pixel_distances
    contact_index = int(valid[np.argmin(pixel_distances[valid])])
    return contact_index, centers, pixel_distances


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--meta-root", type=Path, required=True)
    parser.add_argument("--camera-config-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-tasks", nargs="+", default=sorted(PUSH_TASKS))
    parser.add_argument("--max-episodes-per-task", type=int, default=0)
    parser.add_argument("--flow-mode", choices=("semantic_axis", "eef_trajectory"), default="semantic_axis")
    parser.add_argument("--omit-images", action="store_true")
    args = parser.parse_args()

    include_tasks = set(args.include_tasks)
    unsupported = include_tasks - set(PUSH_TASKS)
    if unsupported:
        parser.error(f"This converter currently supports block-push tasks only: {sorted(unsupported)}")

    task_rows = load_jsonl(args.meta_root / "tasks.jsonl")
    task_by_id = {int(row["task_index"]): str(row["task"]) for row in task_rows}
    episode_rows = {int(row["episode_index"]): row for row in load_jsonl(args.meta_root / "episodes.jsonl")}
    camera_config = yaml.safe_load(
        (args.camera_config_root / "training" / ".hydra" / "merged_config.yaml").read_text()
    )["cameras"]["static"]
    matrix = camera_matrix(camera_config)
    width = int(camera_config["width"])
    height = int(camera_config["height"])

    split_dir = args.output_dir / "training"
    split_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    annotations: list[str] = []
    tasks: list[str] = []
    intervals: list[list[int]] = []
    episode_manifest: list[dict[str, object]] = []

    parquet_paths = sorted(args.lerobot_root.glob("data/chunk-*/episode_*.parquet"))
    for parquet_path in parquet_paths:
        episode_index = int(parquet_path.stem.rsplit("_", 1)[1])
        episode_meta = episode_rows.get(episode_index)
        if episode_meta is None:
            continue
        task_text = str(episode_meta["tasks"][0])
        task = task_text.split(":", 1)[0]
        if task not in include_tasks:
            continue
        if args.max_episodes_per_task > 0 and counts[task] >= args.max_episodes_per_task:
            continue

        table = pq.read_table(parquet_path)
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        frame_indices = np.asarray(table["index"].to_pylist(), dtype=np.int64)
        task_indices = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
        if len(np.unique(task_indices)) != 1:
            raise RuntimeError(f"Multiple tasks in {parquet_path}")
        if task_by_id[int(task_indices[0])].split(":", 1)[0] != task:
            raise RuntimeError(f"Task metadata mismatch in {parquet_path}")
        static_images = decode_images(table["observation.images.top"].to_pylist())
        wrist_images = (
            []
            if args.omit_images
            else decode_images(table["observation.images.wrist"].to_pylist())
        )

        color, semantic_flow = PUSH_TASKS[task]
        contact_index, _centers, pixel_distances = infer_contact(
            static_images, state[:, :3], color, matrix, width, height
        )
        anchor_stop = min(contact_index + 3, len(state))
        contact_anchor = np.median(state[contact_index:anchor_stop, :3], axis=0).astype(np.float32)
        if args.flow_mode == "eef_trajectory":
            tail = np.median(state[max(contact_index, len(state) - 3) :, :3], axis=0)
            delta = tail - contact_anchor
            norm = float(np.linalg.norm(delta))
            flow = semantic_flow if norm < 0.02 else (delta / norm).astype(np.float32)
        else:
            flow = semantic_flow

        for local_index, frame_index in enumerate(frame_indices):
            attraction = (
                contact_anchor - state[local_index, :3]
                if local_index <= contact_index
                else np.zeros(3, dtype=np.float32)
            )
            minimal_g = np.concatenate([attraction, flow]).astype(np.float32)
            values: dict[str, np.ndarray] = {
                "robot_obs": state[local_index],
                "scene_obs": np.zeros(24, dtype=np.float32),
                "rel_actions": action[local_index],
                "minimal_g": minimal_g,
            }
            if not args.omit_images:
                values["rgb_static"] = static_images[local_index]
                values["rgb_gripper"] = wrist_images[local_index]
            np.savez(split_dir / f"episode_{int(frame_index):07d}.npz", **values)

        description = task_text.split(":", 1)[1].strip() if ":" in task_text else task
        annotations.append(description)
        tasks.append(task)
        intervals.append([int(frame_indices[0]), int(frame_indices[-1])])
        counts[task] += 1
        finite_distances = pixel_distances[np.isfinite(pixel_distances)]
        episode_manifest.append(
            {
                "episode_index": episode_index,
                "task": task,
                "frames": int(len(frame_indices)),
                "contact_index": int(contact_index),
                "contact_anchor": contact_anchor.tolist(),
                "flow": flow.tolist(),
                "flow_semantic_cosine": float(np.dot(flow, semantic_flow)),
                "target_eef_pixel_distance_median": (
                    float(np.median(finite_distances)) if len(finite_distances) else None
                ),
            }
        )
        print(
            f"[convert] episode={episode_index} task={task} frames={len(frame_indices)} "
            f"contact={contact_index} count={counts[task]}",
            flush=True,
        )

    if not intervals:
        raise RuntimeError("No matching LeRobot episodes were converted")
    language_dir = split_dir / "lang_annotations"
    language_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        language_dir / "auto_lang_ann.npy",
        {
            "language": {"ann": annotations, "task": tasks},
            "info": {"indx": np.asarray(intervals, dtype=np.int64)},
        },
        allow_pickle=True,
    )
    manifest = {
        "source": str(args.lerobot_root.resolve()),
        "meta_root": str(args.meta_root.resolve()),
        "camera_config_root": str(args.camera_config_root.resolve()),
        "flow_mode": args.flow_mode,
        "images_included": not args.omit_images,
        "episodes": int(len(intervals)),
        "frames": int(sum(end - start + 1 for start, end in intervals)),
        "task_counts": dict(sorted(counts.items())),
        "episode_records": episode_manifest,
    }
    (args.output_dir / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: value for key, value in manifest.items() if key != "episode_records"}, indent=2))


if __name__ == "__main__":
    main()
