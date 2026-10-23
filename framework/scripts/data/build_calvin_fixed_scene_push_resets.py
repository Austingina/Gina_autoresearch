#!/usr/bin/env python3
"""Build deterministic, native Scene-D reset pools for balanced CALVIN push collection."""
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
DIRECTIONS = ("left", "right")
PUSH_TASKS = tuple(f"push_{color}_block_{direction}" for color in COLORS for direction in DIRECTIONS)
RUNTIME_SCENE = "calvin_scene_D"
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


def validate_candidate(
    scene_obs: np.ndarray,
    *,
    x_bounds: tuple[float, float] = (-0.14, 0.14),
    z_bounds: tuple[float, float] = (0.44, 0.49),
    min_xy_separation: float = 0.06,
) -> dict[str, list[float]]:
    """Validate one Scene-D state for use by every object×direction task."""
    scene = np.asarray(scene_obs, dtype=np.float32)
    if scene.shape != (24,):
        raise ValueError(f"Scene-D observation must have shape (24,), got {scene.shape}")
    positions = {name: scene[index].astype(float) for name, index in SCENE_D_OBJECT_SLICES.items()}
    for name, pos in positions.items():
        if not x_bounds[0] <= float(pos[0]) <= x_bounds[1]:
            raise ValueError(f"{name} outside shared bidirectional workspace: x={pos[0]:.4f}")
        if not z_bounds[0] <= float(pos[2]) <= z_bounds[1]:
            raise ValueError(f"{name} has invalid table height: z={pos[2]:.4f}")
    names = list(positions)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            distance = float(np.linalg.norm(positions[first][:2] - positions[second][:2]))
            if distance < min_xy_separation:
                raise ValueError(f"Insufficient object separation: {first}/{second}={distance:.4f}")
    return {name: [round(float(value), 6) for value in pos] for name, pos in positions.items()}


def build_manifest(candidates: list[dict[str, Any]], *, seed: int, count: int) -> dict[str, Any]:
    if count < 1:
        raise ValueError("count must be positive")
    if len(candidates) < count:
        raise ValueError(f"Need {count} valid candidates, got {len(candidates)}")
    resets = []
    for index, candidate in enumerate(candidates[:count]):
        robot_obs = np.asarray(candidate["robot_obs"], dtype=np.float32)
        scene_obs = np.asarray(candidate["scene_obs"], dtype=np.float32)
        if robot_obs.shape != (15,):
            raise ValueError(f"robot_obs must have shape (15,), got {robot_obs.shape}")
        positions = validate_candidate(scene_obs)
        resets.append(
            {
                "reset_id": f"scene_D_native_{index:04d}",
                "scene": RUNTIME_SCENE,
                "runtime_scene": RUNTIME_SCENE,
                "source_kind": "native_runtime_reset",
                "source_sample_index": int(candidate.get("source_sample_index", index)),
                "robot_obs": robot_obs.tolist(),
                "scene_obs": scene_obs.tolist(),
                "object_positions": positions,
                "eligible_tasks": list(PUSH_TASKS),
            }
        )
    return {
        "format": "calvin_fixed_scene_push_resets_v1",
        "runtime_scene": RUNTIME_SCENE,
        "seed": int(seed),
        "count": len(resets),
        "tasks": list(PUSH_TASKS),
        "factor_axes": {"object": list(COLORS), "direction": list(DIRECTIONS)},
        "controlled_crossing": "every reset is eligible for every object-direction cell",
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
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261020)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.calvin_code = args.calvin_code.resolve()
    args.source_split = args.source_split.resolve()
    args.out = args.out.resolve()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite reset pool: {args.out}")
    if args.count < 1 or args.max_samples < args.count:
        raise ValueError("Invalid count/max-samples")

    config_path = args.source_split / ".hydra/merged_config.yaml"
    config_text = config_path.read_text(encoding="utf-8")
    if "name: calvin_scene_D" not in config_text:
        raise ValueError(f"Source split is not configured for {RUNTIME_SCENE}: {config_path}")

    calvin_models = args.calvin_code / "calvin_models"
    calvin_env = args.calvin_code / "calvin_env"
    sys.path[:0] = [calvin_env.as_posix(), calvin_models.as_posix()]
    from calvin_env.envs.play_table_env import get_env  # noqa: PLC0415

    env = get_env(
        args.source_split,
        show_gui=False,
        obs_space={"rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": []},
    )
    env.seed(args.seed)
    env.scene.np_random = env.np_random
    for obj in env.scene.movable_objects:
        obj.np_random = env.np_random

    candidates: list[dict[str, Any]] = []
    sampled = 0
    try:
        for sampled in range(1, args.max_samples + 1):
            obs = env.reset()
            scene_obs = np.asarray(obs["scene_obs"], dtype=np.float32)
            try:
                validate_candidate(scene_obs)
            except ValueError:
                continue
            candidates.append(
                {
                    "source_sample_index": sampled - 1,
                    "robot_obs": np.asarray(obs["robot_obs"], dtype=np.float32),
                    "scene_obs": scene_obs,
                }
            )
            if len(candidates) >= args.count:
                break
    finally:
        env.close()
        env.ownsPhysicsClient = False

    manifest = build_manifest(candidates, seed=args.seed, count=args.count)
    manifest["provenance"] = {
        "builder": Path(__file__).resolve().as_posix(),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "source_split": args.source_split.as_posix(),
        "source_config": config_path.as_posix(),
        "source_config_sha256": sha256_file(config_path),
        "native_samples_examined": sampled,
        "selection": "native Scene-D reset; all objects central enough for both x directions; valid height and pair separation",
    }
    write_json_atomic(args.out, manifest)
    print(json.dumps({"out": args.out.as_posix(), "accepted_resets": args.count, "native_samples_examined": sampled}))


if __name__ == "__main__":
    main()
