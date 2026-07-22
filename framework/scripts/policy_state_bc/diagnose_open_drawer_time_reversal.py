#!/usr/bin/env python3
"""Test whether CALVIN open-drawer demonstrations are physically reversible.

The diagnostic distinguishes two questions:

1. Can reversed relative actions be used directly as synthetic labels?
2. Is the recorded contact path itself feasible in reverse when a feedback
   controller tracks the preceding recorded TCP pose?

Both variants start from the same post-demonstration simulator state and use
CALVIN's official close-drawer task oracle.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

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
sys.path.insert(0, SCRIPT_DIR.as_posix())

import train_eval_state_bc_task_policy as base  # noqa: E402


VARIANTS = ("negated_recorded_rel", "previous_pose_feedback")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def angle_delta(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    delta = target - current
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def pose_tracking_action(current_robot: np.ndarray, target_robot: np.ndarray) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)
    action[:3] = np.clip((target_robot[:3] - current_robot[:3]) / 0.02, -1.0, 1.0)
    action[3:6] = np.clip(angle_delta(target_robot[3:6], current_robot[3:6]) / 0.05, -1.0, 1.0)
    action[-1] = 1.0 if float(target_robot[-1]) >= 0.0 else -1.0
    return action


def load_segment_frames(split_dir: Path, segment: dict[str, Any]) -> dict[int, dict[str, np.ndarray]]:
    start = int(segment["start"])
    end = int(segment["end"])
    frames: dict[int, dict[str, np.ndarray]] = {}
    for index in range(start, end + 2):
        with np.load(base.episode_file(split_dir, index), allow_pickle=True) as data:
            frames[index] = {
                "robot_obs": np.asarray(data["robot_obs"], dtype=np.float32),
                "scene_obs": np.asarray(data["scene_obs"], dtype=np.float32),
                "rel_actions": np.asarray(data["rel_actions"], dtype=np.float32),
            }
    return frames


def select_segments(split_dir: Path, num_segments: int, seed: int) -> list[dict[str, Any]]:
    candidates = [segment for segment in base.load_lang_segments(split_dir) if segment["task"] == "open_drawer"]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected: list[dict[str, Any]] = []
    for segment in candidates:
        start = int(segment["start"])
        end = int(segment["end"])
        if not base.episode_file(split_dir, end + 1).exists():
            continue
        try:
            with np.load(base.episode_file(split_dir, start), allow_pickle=True) as first:
                start_q = float(first["scene_obs"][1])
            with np.load(base.episode_file(split_dir, end + 1), allow_pickle=True) as final:
                final_q = float(final["scene_obs"][1])
        except Exception:
            continue
        if final_q - start_q < 0.12:
            continue
        selected.append(segment)
        if len(selected) >= num_segments:
            break
    if len(selected) < num_segments:
        raise RuntimeError(f"Only found {len(selected)} valid open_drawer segments; requested {num_segments}")
    return selected


def run_reverse_rollout(
    env: Any,
    task_oracle: Any,
    frames: dict[int, dict[str, np.ndarray]],
    segment: dict[str, Any],
    variant: str,
) -> dict[str, Any]:
    start = int(segment["start"])
    end = int(segment["end"])
    final_frame = frames[end + 1]
    env.reset(robot_obs=final_frame["robot_obs"], scene_obs=final_frame["scene_obs"])
    start_info = env.get_info()
    obs = env.get_state_obs()
    start_q = float(np.asarray(obs["scene_obs"])[1])

    q_values = [start_q]
    position_errors: list[float] = []
    orientation_errors: list[float] = []
    success_step: int | None = None
    gripper_commands: Counter[int] = Counter()

    for step, index in enumerate(range(end, start - 1, -1), start=1):
        target_robot = frames[index]["robot_obs"]
        current_robot = np.asarray(obs["robot_obs"], dtype=np.float32)
        if variant == "negated_recorded_rel":
            action = frames[index]["rel_actions"].copy()
            action[:6] *= -1.0
            action[-1] = 1.0 if float(target_robot[-1]) >= 0.0 else -1.0
        elif variant == "previous_pose_feedback":
            action = pose_tracking_action(current_robot, target_robot)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        gripper_commands[int(action[-1])] += 1
        obs, _, _, current_info = env.step(action)
        next_robot = np.asarray(obs["robot_obs"], dtype=np.float32)
        next_scene = np.asarray(obs["scene_obs"], dtype=np.float32)
        q_values.append(float(next_scene[1]))
        position_errors.append(float(np.linalg.norm(next_robot[:3] - target_robot[:3])))
        orientation_errors.append(float(np.linalg.norm(angle_delta(target_robot[3:6], next_robot[3:6]))))
        if success_step is None and task_oracle.get_task_info_for_set(start_info, current_info, {"close_drawer"}):
            success_step = step

    q = np.asarray(q_values, dtype=np.float64)
    return {
        "variant": variant,
        "segment_start": start,
        "segment_end": end,
        "num_actions": end - start + 1,
        "success": success_step is not None,
        "success_step": success_step,
        "drawer_q_start": start_q,
        "drawer_q_end": float(q[-1]),
        "drawer_q_min": float(np.min(q)),
        "drawer_close_delta": float(start_q - q[-1]),
        "drawer_max_close_delta": float(start_q - np.min(q)),
        "position_tracking_rmse": float(np.sqrt(np.mean(np.square(position_errors)))),
        "orientation_tracking_rmse": float(np.sqrt(np.mean(np.square(orientation_errors)))),
        "gripper_commands": {str(key): int(value) for key, value in sorted(gripper_commands.items())},
        "drawer_q_trace": q.tolist(),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [bool(row["success"]) for row in rows]
    output: dict[str, Any] = {
        "count": len(rows),
        "success": int(sum(successes)),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
    }
    for key in [
        "drawer_close_delta",
        "drawer_max_close_delta",
        "position_tracking_rmse",
        "orientation_tracking_rmse",
    ]:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        output[key] = {
            "mean": float(np.mean(values)),
            "p10": float(np.percentile(values, 10)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calvin-code", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-segments", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()

    calvin_code = args.calvin_code.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    split_dir = data_root / "training"
    selected = select_segments(split_dir, args.num_segments, args.seed)

    calvin_models = calvin_code / "calvin_models"
    calvin_env = calvin_code / "calvin_env"
    calvin_agent = calvin_models / "calvin_agent"
    sys.path.insert(0, calvin_models.as_posix())
    sys.path.insert(0, calvin_env.as_posix())
    os.chdir(calvin_agent)

    from calvin_agent.evaluation import evaluate_policy as eval_mod
    import hydra
    from omegaconf import OmegaConf

    task_cfg = OmegaConf.load(calvin_models / "conf/callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    env = eval_mod.make_env(data_root)
    env.get_obs = env.get_state_obs

    rows: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(selected):
        frames = load_segment_frames(split_dir, segment)
        for variant in args.variants:
            result = run_reverse_rollout(env, task_oracle, frames, segment, variant)
            result["segment_index"] = segment_index
            rows.append(result)
            print(
                f"[{segment_index + 1:02d}/{len(selected):02d}] {variant} "
                f"success={int(result['success'])} close={result['drawer_close_delta']:.3f} "
                f"max_close={result['drawer_max_close_delta']:.3f}",
                flush=True,
            )

    by_variant = {
        variant: aggregate([row for row in rows if row["variant"] == variant]) for variant in args.variants
    }
    payload = {
        "protocol": {
            "source_task": "open_drawer",
            "target_task": "close_drawer",
            "post_demo_state_index": "segment_end_plus_one",
            "official_close_threshold": -0.12,
            "num_segments": args.num_segments,
            "seed": args.seed,
            "variants": args.variants,
            "position_action_scale_m": 0.02,
            "orientation_action_scale_rad": 0.05,
            "state_only_observation": True,
        },
        "aggregate": by_variant,
        "rollouts": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "open_drawer_time_reversal.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)
    print(json.dumps(by_variant, indent=2), flush=True)
    # PyBullet/EGL can hang during interpreter teardown on headless servers.
    os._exit(0)


if __name__ == "__main__":
    main()
