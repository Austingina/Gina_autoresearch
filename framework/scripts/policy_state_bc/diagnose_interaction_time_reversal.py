#!/usr/bin/env python3
"""Audit whether source demonstrations are physically valid inverse-task data.

For every SOURCE=TARGET pair, reset CALVIN to the state immediately after a
source demonstration and execute the recorded TCP path backward. The target
task oracle decides whether the inverse interaction is physically valid.
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


def parse_task_pairs(items: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected SOURCE=TARGET, got {item!r}")
        source, target = (part.strip() for part in item.split("=", 1))
        if not source or not target:
            raise ValueError(f"Invalid task pair {item!r}")
        pairs.append((source, target))
    return pairs


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


def select_segments(
    split_dir: Path,
    source_task: str,
    num_segments: int,
    seed: int,
) -> list[dict[str, Any]]:
    candidates = [
        segment
        for segment in base.load_lang_segments(split_dir)
        if str(segment["task"]) == source_task
        and base.episode_file(split_dir, int(segment["end"]) + 1).exists()
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    if len(candidates) < num_segments:
        raise RuntimeError(
            f"Only found {len(candidates)} usable {source_task} segments; requested {num_segments}"
        )
    return candidates[:num_segments]


def run_reverse_rollout(
    env: Any,
    task_oracle: Any,
    frames: dict[int, dict[str, np.ndarray]],
    segment: dict[str, Any],
    target_task: str,
    variant: str,
) -> dict[str, Any]:
    start = int(segment["start"])
    end = int(segment["end"])
    final_frame = frames[end + 1]
    env.reset(robot_obs=final_frame["robot_obs"], scene_obs=final_frame["scene_obs"])
    start_info = env.get_info()
    obs = env.get_state_obs()
    initial_robot = np.asarray(obs["robot_obs"], dtype=np.float32)
    initial_scene = np.asarray(obs["scene_obs"], dtype=np.float32)

    position_errors: list[float] = []
    orientation_errors: list[float] = []
    gripper_commands: Counter[int] = Counter()
    success_step: int | None = None

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
        position_errors.append(float(np.linalg.norm(next_robot[:3] - target_robot[:3])))
        orientation_errors.append(
            float(np.linalg.norm(angle_delta(target_robot[3:6], next_robot[3:6])))
        )
        if success_step is None and task_oracle.get_task_info_for_set(
            start_info, current_info, {target_task}
        ):
            success_step = step

    final_robot = np.asarray(obs["robot_obs"], dtype=np.float32)
    final_scene = np.asarray(obs["scene_obs"], dtype=np.float32)
    return {
        "variant": variant,
        "target_task": target_task,
        "segment_start": start,
        "segment_end": end,
        "num_actions": end - start + 1,
        "success": success_step is not None,
        "success_step": success_step,
        "eef_displacement": (final_robot[:3] - initial_robot[:3]).tolist(),
        "scene_displacement_l2": float(np.linalg.norm(final_scene - initial_scene)),
        "position_tracking_rmse": float(np.sqrt(np.mean(np.square(position_errors)))),
        "orientation_tracking_rmse": float(np.sqrt(np.mean(np.square(orientation_errors)))),
        "gripper_commands": {str(key): int(value) for key, value in sorted(gripper_commands.items())},
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success = np.asarray([bool(row["success"]) for row in rows], dtype=bool)
    position_rmse = np.asarray([row["position_tracking_rmse"] for row in rows], dtype=np.float64)
    return {
        "count": int(len(rows)),
        "success": int(np.sum(success)),
        "success_rate": float(np.mean(success)) if len(success) else 0.0,
        "position_tracking_rmse": {
            "p50": float(np.percentile(position_rmse, 50)),
            "p90": float(np.percentile(position_rmse, 90)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calvin-code", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--task-pairs", nargs="+", required=True)
    parser.add_argument("--num-segments", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()

    task_pairs = parse_task_pairs(args.task_pairs)
    data_root = args.data_root.expanduser().resolve()
    split_dir = data_root / "training"
    calvin_code = args.calvin_code.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    calvin_models = calvin_code / "calvin_models"
    sys.path.insert(0, calvin_models.as_posix())
    sys.path.insert(0, (calvin_code / "calvin_env").as_posix())
    os.chdir(calvin_models / "calvin_agent")

    from calvin_agent.evaluation import evaluate_policy as eval_mod
    import hydra
    from omegaconf import OmegaConf

    task_cfg = OmegaConf.load(calvin_models / "conf/callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    env = eval_mod.make_env(data_root)
    env.get_obs = env.get_state_obs

    rows: list[dict[str, Any]] = []
    for pair_index, (source_task, target_task) in enumerate(task_pairs):
        selected = select_segments(split_dir, source_task, args.num_segments, args.seed)
        for segment_index, segment in enumerate(selected):
            frames = load_segment_frames(split_dir, segment)
            for variant in args.variants:
                result = run_reverse_rollout(
                    env, task_oracle, frames, segment, target_task, variant
                )
                result.update(
                    {
                        "source_task": source_task,
                        "pair_index": pair_index,
                        "segment_index": segment_index,
                    }
                )
                rows.append(result)
                print(
                    f"[{source_task}->{target_task} {segment_index + 1:02d}/{len(selected):02d}] "
                    f"{variant} success={int(result['success'])} "
                    f"tcp_rmse={result['position_tracking_rmse']:.4f}",
                    flush=True,
                )

    aggregate_rows: dict[str, Any] = {}
    for source_task, target_task in task_pairs:
        pair_key = f"{source_task}={target_task}"
        aggregate_rows[pair_key] = {
            variant: aggregate(
                [
                    row
                    for row in rows
                    if row["source_task"] == source_task
                    and row["target_task"] == target_task
                    and row["variant"] == variant
                ]
            )
            for variant in args.variants
        }

    payload = {
        "protocol": {
            "task_pairs": task_pairs,
            "num_segments_per_pair": args.num_segments,
            "seed": args.seed,
            "variants": args.variants,
            "position_action_scale_m": 0.02,
            "orientation_action_scale_rad": 0.05,
            "state_only_observation": True,
        },
        "aggregate": aggregate_rows,
        "rollouts": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "interaction_time_reversal.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)
    print(json.dumps(aggregate_rows, indent=2), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
