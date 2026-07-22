#!/usr/bin/env python3
"""Utilities for CALVIN visual checks and interaction-geometry prototypes."""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


DEFAULT_OBJECT_NAMES = ("block_red", "block_blue", "block_pink")
DISPLAY_OBJECT_NAMES = {
    "block_red": "red_block",
    "block_blue": "blue_block",
    "block_pink": "pink_block",
}
OBJECT_COLORS = {
    "block_red": (220, 60, 60),
    "block_blue": (55, 110, 230),
    "block_pink": (225, 80, 190),
}
DEFAULT_SCENE_SCALAR_NAMES = (
    "base__slide",
    "base__drawer",
    "base__button",
    "base__switch",
    "lightbulb",
    "led",
)
FRAME_RE = re.compile(r"episode_(\d+)\.npz$")


@dataclass(frozen=True)
class AnnotationEntry:
    split: str
    index: int
    start: int
    end: int
    task: str
    ann: str

    @property
    def length(self) -> int:
        return int(self.end) - int(self.start) + 1


def repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[3]


def default_dataset_root() -> Path:
    return Path("/mnt/nvme33n1/wuk_code/calvin_g/data/calvin_debug_dataset")


def timestamp() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_hydra_config(dataset_root: Path, split: str = "training") -> dict[str, Any]:
    candidates = [
        dataset_root / split / ".hydra" / "merged_config.yaml",
        dataset_root / split / ".hydra" / "config.yaml",
        dataset_root / ".hydra" / "merged_config.yaml",
        dataset_root / ".hydra" / "config.yaml",
    ]
    for path in candidates:
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            return data or {}
    return {}


def scene_schema_from_dataset(dataset_root: Path, split: str = "training") -> dict[str, Any]:
    cfg = load_hydra_config(dataset_root, split)
    scene = cfg.get("scene", {}) if isinstance(cfg, dict) else {}
    fixed_objects = scene.get("objects", {}).get("fixed_objects", {}) if isinstance(scene, dict) else {}
    movable_objects = scene.get("objects", {}).get("movable_objects", {}) if isinstance(scene, dict) else {}
    doors: list[str] = []
    buttons: list[str] = []
    switches: list[str] = []
    lights: list[str] = []
    for _, fixed_cfg in fixed_objects.items():
        joints = fixed_cfg.get("joints", {}) or {}
        doors.extend(list(joints.keys()))
        buttons.extend(list((fixed_cfg.get("buttons", {}) or {}).keys()))
        switches.extend(list((fixed_cfg.get("switches", {}) or {}).keys()))
        lights.extend(list((fixed_cfg.get("lights", {}) or {}).keys()))
    object_names = list(movable_objects.keys()) or list(DEFAULT_OBJECT_NAMES)
    scalar_names = doors + buttons + switches + lights
    if not scalar_names:
        scalar_names = list(DEFAULT_SCENE_SCALAR_NAMES)
    return {
        "scene_name": scene.get("name", "unknown") if isinstance(scene, dict) else "unknown",
        "scalar_names": scalar_names,
        "object_names": object_names,
        "source": "hydra" if cfg else "default",
    }


def object_display_name(name: str) -> str:
    return DISPLAY_OBJECT_NAMES.get(name, name)


def object_slice_map(object_names: Iterable[str], scalar_dim: int = 6) -> dict[str, slice]:
    return {
        name: slice(scalar_dim + i * 6, scalar_dim + (i + 1) * 6)
        for i, name in enumerate(object_names)
    }


def frame_id_from_path(path: Path) -> int:
    m = FRAME_RE.search(path.name)
    if not m:
        raise ValueError(f"Not a CALVIN episode frame file: {path}")
    return int(m.group(1))


def frame_file(dataset_root: Path, split: str, frame_id: int) -> Path:
    return dataset_root / split / f"episode_{frame_id:07d}.npz"


def load_frame(dataset_root: Path, split: str, frame_id: int) -> dict[str, np.ndarray]:
    path = frame_file(dataset_root, split, frame_id)
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def list_frame_ids(dataset_root: Path, split: str) -> list[int]:
    files = sorted((dataset_root / split).glob("episode_*.npz"))
    return [frame_id_from_path(p) for p in files]


def load_annotations(dataset_root: Path, split: str) -> list[AnnotationEntry]:
    split_dir = dataset_root / split
    candidates = [
        split_dir / "lang_annotations" / "auto_lang_ann.npy",
        split_dir / "lang_paraphrase-MiniLM-L3-v2" / "auto_lang_ann.npy",
    ]
    ann_path = next((p for p in candidates if p.exists()), None)
    if ann_path is None:
        raise FileNotFoundError(
            "Could not find auto_lang_ann.npy under lang_annotations or lang_paraphrase-MiniLM-L3-v2"
        )
    data = np.load(ann_path, allow_pickle=True).item()
    language = data["language"]
    info = data["info"]
    tasks = list(language["task"])
    anns = list(language["ann"])
    ranges = list(info["indx"])
    entries: list[AnnotationEntry] = []
    for i, (task, ann, index_range) in enumerate(zip(tasks, anns, ranges)):
        start, end = int(index_range[0]), int(index_range[1])
        entries.append(
            AnnotationEntry(
                split=split,
                index=i,
                start=start,
                end=end,
                task=str(task),
                ann=str(ann),
            )
        )
    return entries


def annotation_to_dict(entry: AnnotationEntry) -> dict[str, Any]:
    return {
        "split": entry.split,
        "index": entry.index,
        "start": entry.start,
        "end": entry.end,
        "length": entry.length,
        "task": entry.task,
        "ann": entry.ann,
    }


def select_annotation_entries(
    entries: list[AnnotationEntry],
    *,
    task: str | None = None,
    indices: list[int] | None = None,
    n: int = 4,
    seed: int = 0,
) -> list[AnnotationEntry]:
    filtered = entries
    if task:
        filtered = [e for e in filtered if task in e.task]
    if indices:
        by_index = {e.index: e for e in filtered}
        return [by_index[i] for i in indices if i in by_index]
    rng = random.Random(seed)
    if len(filtered) <= n:
        return list(filtered)
    return rng.sample(filtered, n)


def sample_frame_ids(start: int, end: int, max_frames: int, stride: int = 1) -> list[int]:
    ids = list(range(int(start), int(end) + 1, max(1, int(stride))))
    if max_frames and len(ids) > max_frames:
        xs = np.linspace(0, len(ids) - 1, max_frames)
        ids = [ids[int(round(x))] for x in xs]
    return ids


def normalize_uint8(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    lo, hi = np.nanpercentile(arr, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or math.isclose(float(lo), float(hi)):
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr) + 1e-6)
    arr = (arr - lo) / max(hi - lo, 1e-6)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def depth_to_rgb(depth: np.ndarray) -> Image.Image:
    gray = normalize_uint8(depth)
    # Lightweight blue-yellow colormap without requiring matplotlib.
    t = gray.astype(np.float32) / 255.0
    rgb = np.stack(
        [
            np.clip(255.0 * t, 0, 255),
            np.clip(255.0 * (1.0 - np.abs(t - 0.5) * 1.4), 0, 255),
            np.clip(255.0 * (1.0 - t), 0, 255),
        ],
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(rgb)


def rgb_to_image(rgb: np.ndarray, size: tuple[int, int] = (200, 200)) -> Image.Image:
    img = Image.fromarray(normalize_uint8(rgb[..., :3]))
    return img.resize(size, Image.Resampling.BILINEAR)


def text_panel(lines: list[str], size: tuple[int, int], bg: tuple[int, int, int] = (248, 248, 248)) -> Image.Image:
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    x, y = 8, 8
    for line in lines:
        draw.text((x, y), str(line)[:96], fill=(20, 20, 20), font=font)
        y += 13
        if y > size[1] - 12:
            break
    return img


def hstack(images: list[Image.Image], bg: tuple[int, int, int] = (255, 255, 255), pad: int = 6) -> Image.Image:
    h = max(im.height for im in images)
    w = sum(im.width for im in images) + pad * (len(images) - 1)
    out = Image.new("RGB", (w, h), bg)
    x = 0
    for im in images:
        out.paste(im.convert("RGB"), (x, 0))
        x += im.width + pad
    return out


def vstack(images: list[Image.Image], bg: tuple[int, int, int] = (255, 255, 255), pad: int = 8) -> Image.Image:
    w = max(im.width for im in images)
    h = sum(im.height for im in images) + pad * (len(images) - 1)
    out = Image.new("RGB", (w, h), bg)
    y = 0
    for im in images:
        out.paste(im.convert("RGB"), (0, y))
        y += im.height + pad
    return out


def add_label(img: Image.Image, label: str) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, out.width, 16), fill=(0, 0, 0))
    draw.text((4, 3), label, fill=(255, 255, 255), font=font)
    return out


def parse_scene_obs(
    scene_obs: np.ndarray,
    *,
    scalar_names: Iterable[str] = DEFAULT_SCENE_SCALAR_NAMES,
    object_names: Iterable[str] = DEFAULT_OBJECT_NAMES,
) -> dict[str, Any]:
    scene = np.asarray(scene_obs, dtype=np.float32)
    scalar_names = list(scalar_names)
    object_names = list(object_names)
    scalars = {name: float(scene[i]) for i, name in enumerate(scalar_names)}
    objects: dict[str, dict[str, list[float]]] = {}
    for name, sl in object_slice_map(object_names, len(scalar_names)).items():
        vals = scene[sl]
        objects[name] = {
            "pos": vals[:3].astype(float).tolist(),
            "orn": vals[3:6].astype(float).tolist(),
        }
    return {"scalars": scalars, "objects": objects}


def compute_g_features(
    frame: dict[str, np.ndarray],
    *,
    contact_threshold: float = 0.10,
    scalar_names: Iterable[str] = DEFAULT_SCENE_SCALAR_NAMES,
    object_names: Iterable[str] = DEFAULT_OBJECT_NAMES,
) -> dict[str, Any]:
    robot = np.asarray(frame["robot_obs"], dtype=np.float32)
    scene = np.asarray(frame["scene_obs"], dtype=np.float32)
    scalar_names = list(scalar_names)
    object_names = list(object_names)
    eef_pos = robot[:3]
    eef_orn = robot[3:6]
    gripper = float(robot[6]) if robot.shape[0] > 6 else float("nan")
    object_pos = []
    object_orn = []
    rel = []
    dists = []
    for name in object_names:
        vals = scene[object_slice_map(object_names, len(scalar_names))[name]]
        pos = vals[:3].astype(np.float32)
        orn = vals[3:6].astype(np.float32)
        object_pos.append(pos)
        object_orn.append(orn)
        delta = pos - eef_pos
        rel.append(delta)
        dists.append(float(np.linalg.norm(delta)))
    distances = np.asarray(dists, dtype=np.float32)
    near_contact = distances < float(contact_threshold)
    nearest_idx = int(np.argmin(distances))
    return {
        "eef_pos": eef_pos.astype(np.float32),
        "eef_orn": eef_orn.astype(np.float32),
        "gripper": np.asarray([gripper], dtype=np.float32),
        "scene_scalars": scene[: len(scalar_names)].astype(np.float32),
        "object_pos": np.stack(object_pos, axis=0).astype(np.float32),
        "object_orn": np.stack(object_orn, axis=0).astype(np.float32),
        "rel_object_to_eef": np.stack(rel, axis=0).astype(np.float32),
        "dist_object_to_eef": distances,
        "near_contact": near_contact.astype(np.float32),
        "nearest_object_index": np.asarray([nearest_idx], dtype=np.int64),
        "nearest_object_name": object_names[nearest_idx],
    }


def make_raw_frame_panel(
    frame: dict[str, np.ndarray],
    *,
    title: str,
    text_lines: list[str],
    tile_size: tuple[int, int] = (200, 200),
) -> Image.Image:
    static = add_label(rgb_to_image(frame["rgb_static"], tile_size), "rgb_static")
    gripper = add_label(rgb_to_image(frame["rgb_gripper"], tile_size), "rgb_gripper")
    d_static = add_label(depth_to_rgb(frame["depth_static"]).resize(tile_size), "depth_static")
    d_gripper = add_label(depth_to_rgb(frame["depth_gripper"]).resize(tile_size), "depth_gripper")
    text = text_panel([title] + text_lines, (360, tile_size[1]))
    return hstack([static, gripper, d_static, d_gripper, text])


def task_family(task: str) -> str:
    if task.startswith("rotate"):
        return "rotate"
    if task.startswith("push_into"):
        return "push_into"
    if task.startswith("push"):
        return "push"
    if task.startswith("lift"):
        return "lift"
    if task.startswith("place"):
        return "place"
    if task.startswith("stack") or task.startswith("unstack"):
        return "stack"
    if "drawer" in task:
        return "drawer"
    if "slider" in task:
        return "slider"
    if "light" in task or "led" in task:
        return "light"
    return "other"


def task_object_color(task: str) -> str:
    for color in ("red", "blue", "pink"):
        if color in task:
            return color
    return "none"
