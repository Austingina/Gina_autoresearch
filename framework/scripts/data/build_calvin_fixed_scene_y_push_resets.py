#!/usr/bin/env python3
"""Build Scene-D-native resets for paired world-Y forward/backward push tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

COLORS = ("red", "blue", "pink")
DIRECTIONS = ("forward", "backward")
RUNTIME_SCENE = "calvin_scene_D"
TARGET_Y_BOUNDS = (-0.050, -0.030)
TARGET_X_BOUNDS = (0.030, 0.120)
X_BOUNDS = (-0.14, 0.14)
Z_BOUNDS = (0.44, 0.49)
MIN_XY_SEPARATION = 0.06
MIN_PUSH_CORRIDOR_X_SEPARATION = 0.075
SCENE_D_OBJECT_SLICES = {
    "block_red": slice(6, 9),
    "block_blue": slice(12, 15),
    "block_pink": slice(18, 21),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_candidate(scene_obs: np.ndarray, *, target_color: str) -> dict[str, list[float]]:
    if target_color not in COLORS:
        raise ValueError(f"Unknown target color: {target_color}")
    scene = np.asarray(scene_obs, dtype=np.float32)
    if scene.shape != (24,):
        raise ValueError(f"Scene-D observation must have shape (24,), got {scene.shape}")
    positions = {name: scene[index].astype(float) for name, index in SCENE_D_OBJECT_SLICES.items()}
    for name, pos in positions.items():
        if not X_BOUNDS[0] <= float(pos[0]) <= X_BOUNDS[1]:
            raise ValueError(f"{name} outside X workspace: {pos[0]:.4f}")
        if not Z_BOUNDS[0] <= float(pos[2]) <= Z_BOUNDS[1]:
            raise ValueError(f"{name} invalid table height: {pos[2]:.4f}")
    target = positions[f"block_{target_color}"]
    if not TARGET_X_BOUNDS[0] <= float(target[0]) <= TARGET_X_BOUNDS[1]:
        raise ValueError(f"block_{target_color} outside reachable X band: x={target[0]:.4f}")
    if not TARGET_Y_BOUNDS[0] <= float(target[1]) <= TARGET_Y_BOUNDS[1]:
        raise ValueError(f"block_{target_color} lacks bidirectional Y margin: y={target[1]:.4f}")
    for name, pos in positions.items():
        if name != f"block_{target_color}" and abs(float(pos[0] - target[0])) < MIN_PUSH_CORRIDOR_X_SEPARATION:
            raise ValueError(f"{name} obstructs target Y-push corridor")
    names = list(positions)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            distance = float(np.linalg.norm(positions[first][:2] - positions[second][:2]))
            if distance < MIN_XY_SEPARATION:
                raise ValueError(f"Insufficient object separation: {first}/{second}={distance:.4f}")
    return {name: [round(float(value), 6) for value in pos] for name, pos in positions.items()}


def build_manifest(
    candidates: list[dict[str, Any]],
    *,
    seed: int,
    count_per_color: int,
    skip_per_color: int = 0,
    id_namespace: str = "",
) -> dict[str, Any]:
    if count_per_color < 1:
        raise ValueError("count_per_color must be positive")
    if skip_per_color < 0:
        raise ValueError("skip_per_color must be non-negative")
    if id_namespace and not all(char.isalnum() or char in "_-" for char in id_namespace):
        raise ValueError("id_namespace must contain only letters, digits, underscore, or hyphen")
    grouped = {color: [] for color in COLORS}
    required_per_color = skip_per_color + count_per_color
    for candidate in candidates:
        color = str(candidate["target_color"])
        if color in grouped and len(grouped[color]) < required_per_color:
            grouped[color].append(candidate)
    missing = {color: required_per_color - len(rows) for color, rows in grouped.items() if len(rows) < required_per_color}
    if missing:
        raise ValueError(f"Insufficient target-specific candidates: {missing}")
    resets = []
    for color in COLORS:
        selected = grouped[color][skip_per_color:required_per_color]
        for index, candidate in enumerate(selected):
            robot_obs = np.asarray(candidate["robot_obs"], dtype=np.float32)
            scene_obs = np.asarray(candidate["scene_obs"], dtype=np.float32)
            if robot_obs.shape != (15,):
                raise ValueError(f"robot_obs must have shape (15,), got {robot_obs.shape}")
            positions = validate_candidate(scene_obs, target_color=color)
            reset_id_prefix = f"{id_namespace}_" if id_namespace else ""
            resets.append({
                "reset_id": f"scene_D_native_y_{reset_id_prefix}{color}_{index:04d}",
                "scene": RUNTIME_SCENE,
                "runtime_scene": RUNTIME_SCENE,
                "source_kind": "native_runtime_reset",
                "source_sample_index": int(candidate.get("source_sample_index", index)),
                "target_color": color,
                "robot_obs": robot_obs.tolist(),
                "scene_obs": scene_obs.tolist(),
                "object_positions": positions,
                "eligible_tasks": [f"push_{color}_block_forward", f"push_{color}_block_backward"],
            })
    tasks = [f"push_{color}_block_{direction}" for color in COLORS for direction in DIRECTIONS]
    return {
        "format": "calvin_fixed_scene_push_resets_v1",
        "runtime_scene": RUNTIME_SCENE,
        "seed": int(seed),
        "id_namespace": id_namespace or None,
        "skip_per_color": int(skip_per_color),
        "count": len(resets),
        "count_per_color": count_per_color,
        "tasks": tasks,
        "factor_axes": {"object": list(COLORS), "direction": list(DIRECTIONS)},
        "direction_world_flows": {"forward": [0, 1, 0], "backward": [0, -1, 0]},
        "controlled_crossing": "each target-specific reset is shared by that object's forward/backward cells",
        "resets": resets,
    }


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{path.name}.tmp"
    with open(temp, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    os.replace(temp, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calvin-code", type=Path, required=True)
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count-per-color", type=int, default=24)
    parser.add_argument("--skip-per-color", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--id-namespace", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.calvin_code = args.calvin_code.resolve()
    args.source_split = args.source_split.resolve()
    args.out = args.out.resolve()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite reset pool: {args.out}")
    if args.count_per_color < 1 or args.skip_per_color < 0 or args.max_samples < args.count_per_color + args.skip_per_color:
        raise ValueError("Invalid count/max-samples")
    config_path = args.source_split / ".hydra/merged_config.yaml"
    config_text = config_path.read_text(encoding="utf-8")
    if "name: calvin_scene_D" not in config_text:
        raise ValueError(f"Source split is not configured for {RUNTIME_SCENE}: {config_path}")
    calvin_models = args.calvin_code / "calvin_models"
    calvin_env = args.calvin_code / "calvin_env"
    sys.path[:0] = [calvin_env.as_posix(), calvin_models.as_posix()]
    from calvin_env.envs.play_table_env import get_env  # noqa: PLC0415

    env = get_env(args.source_split, show_gui=False, obs_space={"rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": []})
    env.seed(args.seed)
    env.scene.np_random = env.np_random
    for obj in env.scene.movable_objects:
        obj.np_random = env.np_random
    candidates: list[dict[str, Any]] = []
    counts = {color: 0 for color in COLORS}
    sampled = 0
    try:
        for sampled in range(1, args.max_samples + 1):
            obs = env.reset()
            scene_obs = np.asarray(obs["scene_obs"], dtype=np.float32)
            for color in COLORS:
                if counts[color] >= args.count_per_color + args.skip_per_color:
                    continue
                try:
                    validate_candidate(scene_obs, target_color=color)
                except ValueError:
                    continue
                candidates.append({
                    "target_color": color,
                    "source_sample_index": sampled - 1,
                    "robot_obs": np.asarray(obs["robot_obs"], dtype=np.float32),
                    "scene_obs": scene_obs,
                })
                counts[color] += 1
            if all(value >= args.count_per_color + args.skip_per_color for value in counts.values()):
                break
    finally:
        env.close()
        env.ownsPhysicsClient = False

    manifest = build_manifest(
        candidates,
        seed=args.seed,
        count_per_color=args.count_per_color,
        skip_per_color=args.skip_per_color,
        id_namespace=args.id_namespace,
    )
    manifest["provenance"] = {
        "builder": Path(__file__).resolve().as_posix(),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "source_split": args.source_split.as_posix(),
        "source_config": config_path.as_posix(),
        "source_config_sha256": sha256_file(config_path),
        "native_samples_examined": sampled,
        "selection": "native Scene-D reset; target central in Y for +/-0.12m travel; valid height and pair separation",
    }
    write_json_atomic(args.out, manifest)
    print(json.dumps({"out": args.out.as_posix(), "accepted_resets": len(manifest["resets"]), "per_color": counts, "native_samples_examined": sampled}))


if __name__ == "__main__":
    main()
