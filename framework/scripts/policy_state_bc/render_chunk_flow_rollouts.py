#!/usr/bin/env python3
"""Render CALVIN rollout videos for chunk-BC / chunk-FM policies."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
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

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, SCRIPT_DIR.as_posix())

import train_eval_state_bc_task_policy as base  # noqa: E402
import train_eval_state_chunk_flow_policy as chunk_pol  # noqa: E402


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def parse_policy_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        path = Path(spec)
        return path.parent.parent.name, path
    alias, path = spec.split("=", 1)
    alias = alias.strip()
    if not alias:
        raise ValueError(f"Empty policy alias in {spec!r}")
    return alias, Path(path)


def draw_label(rgb: np.ndarray, lines: list[str]) -> np.ndarray:
    bgr = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    y = 22
    for line in lines:
        cv2.putText(bgr, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(bgr, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        raise RuntimeError(f"No frames to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calvin-code", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", required=True, help="alias=/path/to/policy.pt entries")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--rollout-indices", nargs="+", type=int, required=True)
    parser.add_argument("--candidate-rollouts-per-task", type=int, default=25)
    parser.add_argument("--ep-len", type=int, default=160)
    parser.add_argument("--scan-sequences", type=int, default=5000)
    parser.add_argument("--render-stride", type=int, default=2)
    parser.add_argument("--post-success-steps", type=int, default=12)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--g-scale", type=float, default=1.0)
    parser.add_argument("--query-g-control", default="normal")
    parser.add_argument("--language-control", default="normal")
    parser.add_argument("--action-execution-horizon", type=int, default=1)
    args = parser.parse_args()

    launch_cwd = Path.cwd().resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else (launch_cwd / path).resolve()

    args.calvin_code = resolve(args.calvin_code)
    args.data_root = resolve(args.data_root)
    args.out_dir = resolve(args.out_dir)
    policies = [(alias, resolve(path)) for alias, path in map(parse_policy_spec, args.policies)]
    max_rollout_idx = max(args.rollout_indices)
    candidate_rollouts_per_task = max(args.candidate_rollouts_per_task, max_rollout_idx + 1)

    calvin_models = args.calvin_code / "calvin_models"
    calvin_env = args.calvin_code / "calvin_env"
    calvin_agent = calvin_models / "calvin_agent"
    sys.path.insert(0, calvin_models.as_posix())
    sys.path.insert(0, calvin_env.as_posix())
    os.chdir(calvin_agent)

    from calvin_agent.evaluation import evaluate_policy as eval_mod
    from calvin_agent.evaluation.multistep_sequences import get_sequences
    import hydra
    from omegaconf import OmegaConf

    text_to_task, _task_to_text = base.load_eval_annotation_maps(args.calvin_code)
    segments = base.load_lang_segments(args.data_root / "training")
    task_to_id = base.build_task_index(segments, text_to_task)

    buckets: dict[str, list[Any]] = {task: [] for task in args.tasks}
    for initial_state, seq in get_sequences(max(args.scan_sequences, candidate_rollouts_per_task * 10)):
        first_task = seq[0]
        if first_task in buckets and len(buckets[first_task]) < candidate_rollouts_per_task:
            buckets[first_task].append(initial_state)
        if all(len(v) >= candidate_rollouts_per_task for v in buckets.values()):
            break
    missing = {task: len(states) for task, states in buckets.items() if len(states) <= max_rollout_idx}
    if missing:
        raise RuntimeError(f"Could not find enough initial states: {missing}")

    task_cfg = OmegaConf.load(calvin_models / "conf/callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(calvin_models / "conf/annotations/new_playtable_validation.yaml")
    env = eval_mod.make_env(args.data_root)

    summary: dict[str, Any] = {
        "data_root": args.data_root,
        "policies": [{"alias": alias, "path": path} for alias, path in policies],
        "tasks": args.tasks,
        "rollout_indices": args.rollout_indices,
        "g_scale": args.g_scale,
        "query_g_control": args.query_g_control,
        "language_control": args.language_control,
        "action_execution_horizon": args.action_execution_horizon,
        "videos": [],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for alias, ckpt_path in policies:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing policy checkpoint: {ckpt_path}")
        model = chunk_pol.ChunkFlowPolicy(
            ckpt_path,
            task_to_id,
            text_to_task,
            device=args.device,
            g_scale=args.g_scale,
            query_g_control=args.query_g_control,
            language_control=args.language_control,
            action_execution_horizon=args.action_execution_horizon,
        )
        for task in args.tasks:
            lang_annotation = val_annotations[task][0]
            for rollout_idx in args.rollout_indices:
                initial_state = buckets[task][rollout_idx]
                robot_obs, scene_obs = eval_mod.get_env_state_for_initial_condition(initial_state)
                env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
                obs = env.get_obs()
                start_info = env.get_info()
                model.reset()
                frames: list[np.ndarray] = []
                success = False
                success_step: int | None = None
                for step in range(args.ep_len):
                    action = model.step(obs, lang_annotation)
                    obs, _, _, current_info = env.step(action)
                    current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {task})
                    if len(current_task_info) > 0 and not success:
                        success = True
                        success_step = step
                    if step % max(1, args.render_stride) == 0 or step == args.ep_len - 1:
                        frame = env.render(mode="rgb_array")
                        frames.append(
                            draw_label(
                                frame,
                                [
                                    f"{alias} | {task} | rollout {rollout_idx}",
                                    f"G={args.query_g_control} | exec={args.action_execution_horizon}",
                                    f"step {step:03d} | success {int(success)}",
                                ],
                            )
                        )
                    if success and success_step is not None and step >= success_step + args.post_success_steps:
                        break
                video_path = args.out_dir / alias / task / f"rollout_{rollout_idx:02d}_success_{int(success)}.mp4"
                write_mp4(video_path, frames, args.fps)
                record = {
                    "alias": alias,
                    "checkpoint": ckpt_path,
                    "task": task,
                    "rollout_idx": rollout_idx,
                    "success": bool(success),
                    "success_step": success_step,
                    "frames": len(frames),
                    "video": video_path,
                }
                summary["videos"].append(record)
                print(f"[video] {alias} {task} rollout={rollout_idx} success={int(success)} path={video_path}", flush=True)

    with open(args.out_dir / "video_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=json_default)
    print(json.dumps(summary, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
