#!/usr/bin/env python3
"""Render CALVIN state-BC rollout videos for qualitative inspection."""

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

import train_eval_state_bc_task_policy as pol  # noqa: E402


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


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


def load_demo_targets(path: Path | None) -> dict[str, dict[str, np.ndarray]]:
    if path is None or not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in data.files:
        if key.endswith("__starts"):
            task = key[: -len("__starts")]
            targets_key = f"{task}__targets"
            if targets_key in data:
                out[task] = {"starts": data[key].astype(np.float32), "targets": data[targets_key].astype(np.float32)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calvin-code", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--rollouts-per-task", type=int, default=1)
    parser.add_argument("--candidate-rollouts-per-task", type=int, default=0)
    parser.add_argument(
        "--save-rollout-indices",
        type=int,
        nargs="*",
        default=None,
        help="Only write videos for these rollout indices after scanning candidates.",
    )
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--ep-len", type=int, default=160)
    parser.add_argument("--scan-sequences", type=int, default=5000)
    parser.add_argument("--render-stride", type=int, default=2)
    parser.add_argument("--post-success-steps", type=int, default=12)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.calvin_code = args.calvin_code.resolve()
    args.data_root = args.data_root.resolve()
    args.checkpoint_root = args.checkpoint_root.resolve()
    args.out_dir = args.out_dir.resolve()
    save_rollout_indices = set(args.save_rollout_indices) if args.save_rollout_indices is not None else None

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

    text_to_task, _task_to_text = pol.load_eval_annotation_maps(args.calvin_code)
    segments = pol.load_lang_segments(args.data_root / "training")
    task_to_id = pol.build_task_index(segments, text_to_task)
    demos = load_demo_targets(args.checkpoint_root / "demo_oracle_targets.npz")

    candidate_rollouts_per_task = args.candidate_rollouts_per_task or args.rollouts_per_task
    buckets: dict[str, list[Any]] = {task: [] for task in args.tasks}
    for initial_state, seq in get_sequences(args.scan_sequences):
        first_task = seq[0]
        if first_task in buckets and len(buckets[first_task]) < candidate_rollouts_per_task:
            buckets[first_task].append(initial_state)
        if all(len(v) >= candidate_rollouts_per_task for v in buckets.values()):
            break
    missing = {task: len(states) for task, states in buckets.items() if len(states) < candidate_rollouts_per_task}
    if missing:
        raise RuntimeError(f"Could not find enough initial states: {missing}")

    task_cfg = OmegaConf.load(calvin_models / "conf/callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(calvin_models / "conf/annotations/new_playtable_validation.yaml")
    env = eval_mod.make_env(args.data_root)

    summary = {"checkpoint_root": args.checkpoint_root, "videos": [], "attempts": []}
    for mode in args.modes:
        ckpt_path = args.checkpoint_root / mode / "policy.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {mode}: {ckpt_path}")
        model = pol.StateBCPolicy(ckpt_path, task_to_id, text_to_task, demos, device=args.device)
        for task in args.tasks:
            lang_annotation = val_annotations[task][0]
            saved_for_task = 0
            for rollout_idx, initial_state in enumerate(buckets[task]):
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
                        lines = [
                            f"{mode} | {task} | rollout {rollout_idx}",
                            f"step {step:03d} | success {int(success)}",
                        ]
                        frames.append(draw_label(frame, lines))
                    if success and success_step is not None and step >= success_step + args.post_success_steps:
                        break
                attempt_record = {
                    "mode": mode,
                    "task": task,
                    "rollout_idx": rollout_idx,
                    "success": bool(success),
                    "success_step": success_step,
                    "frames": len(frames),
                    "saved": False,
                }
                summary["attempts"].append(attempt_record)
                if args.success_only and not success:
                    print(f"[skip] {mode} {task} rollout={rollout_idx} success=0", flush=True)
                    continue
                if save_rollout_indices is not None and rollout_idx not in save_rollout_indices:
                    continue
                if saved_for_task >= args.rollouts_per_task:
                    continue
                video_path = args.out_dir / mode / task / f"rollout_{rollout_idx:02d}_success_{int(success)}.mp4"
                write_mp4(video_path, frames, args.fps)
                attempt_record["saved"] = True
                attempt_record["video"] = video_path
                saved_for_task += 1
                summary["videos"].append(
                    {
                        "mode": mode,
                        "task": task,
                        "rollout_idx": rollout_idx,
                        "success": bool(success),
                        "success_step": success_step,
                        "frames": len(frames),
                        "video": video_path,
                    }
                )
                print(f"[video] {mode} {task} rollout={rollout_idx} success={int(success)} path={video_path}", flush=True)
                if args.success_only and saved_for_task >= args.rollouts_per_task:
                    break

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "video_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=json_default)
    print(json.dumps(summary, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
