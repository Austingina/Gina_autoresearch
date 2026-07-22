#!/usr/bin/env python3
"""Evaluate visual CALVIN policies from demonstration starts in source scenes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from typing import Any
import zipfile

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, SCRIPT_DIR.as_posix())

import train_eval_state_bc_task_policy as base  # noqa: E402
import train_eval_state_chunk_flow_policy as policy  # noqa: E402


def load_scene_ranges(source_archive: Path) -> tuple[dict[str, tuple[int, int]], str]:
    with zipfile.ZipFile(source_archive) as archive:
        members = [name for name in archive.namelist() if name.endswith("/training/scene_info.npy")]
        if len(members) != 1:
            raise RuntimeError(f"Expected one training/scene_info.npy in {source_archive}, got {members}")
        scene_info = np.load(BytesIO(archive.read(members[0])), allow_pickle=True).item()
    archive_root = members[0][: -len("training/scene_info.npy")]
    ranges = {str(scene): (int(bounds[0]), int(bounds[1])) for scene, bounds in scene_info.items()}
    return ranges, archive_root


def scene_for_frame(frame: int, scene_ranges: dict[str, tuple[int, int]]) -> str:
    matches = [scene for scene, (start, end) in scene_ranges.items() if start <= frame <= end]
    if len(matches) != 1:
        raise RuntimeError(f"Frame {frame} maps to source scenes {matches}")
    return matches[0]


def evenly_spaced(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(items) < count:
        raise RuntimeError(f"Requested {count} starts from only {len(items)} segments")
    indices = np.linspace(0, len(items) - 1, num=count, dtype=np.int64)
    return [items[int(index)] for index in indices]


def make_scene_env(calvin_code: Path, eval_data_root: Path, scene: str) -> Any:
    import hydra
    from omegaconf import OmegaConf

    render_conf = OmegaConf.load(eval_data_root / "validation/.hydra/merged_config.yaml")
    scene_conf = OmegaConf.load(calvin_code / f"calvin_env/conf/scene/{scene}.yaml")
    render_conf.scene = scene_conf
    for camera in set(render_conf.cameras.keys()) - {"static", "gripper"}:
        del render_conf.cameras[camera]
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(".")
    return hydra.utils.instantiate(render_conf.env, show_gui=False, use_vr=False, use_scene_info=True)


def save_progress(path: Path, signature: dict[str, Any], results: list[dict[str, Any]]) -> None:
    payload = {**signature, "results": results}
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=base.json_default)
    os.replace(tmp_path, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calvin-code", type=Path, required=True)
    parser.add_argument("--source-data-root", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--eval-data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-meta", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "push_blue_block_right",
            "push_pink_block_left",
            "push_pink_block_right",
            "push_red_block_left",
        ],
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Source scenes to evaluate; defaults to every scene in scene_info.npy.",
    )
    parser.add_argument("--num-per-task-scene", type=int, default=25)
    parser.add_argument("--ep-len", type=int, default=160)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--g-scale", type=float, default=1.0)
    parser.add_argument("--query-g-control", default="normal")
    parser.add_argument("--language-control", default="normal")
    parser.add_argument("--action-execution-horizon", type=int, default=1)
    parser.add_argument(
        "--audit-reset-visuals",
        action="store_true",
        help="Compare reset camera observations with the matching stored demonstration frame.",
    )
    args = parser.parse_args()

    launch_cwd = Path.cwd().resolve()

    def resolve(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (launch_cwd / path).resolve()

    for field in (
        "calvin_code",
        "source_data_root",
        "source_archive",
        "eval_data_root",
        "checkpoint",
        "run_meta",
        "out_dir",
    ):
        setattr(args, field, resolve(getattr(args, field)))

    calvin_models = args.calvin_code / "calvin_models"
    calvin_env = args.calvin_code / "calvin_env"
    calvin_agent = calvin_models / "calvin_agent"
    sys.path.insert(0, calvin_models.as_posix())
    sys.path.insert(0, calvin_env.as_posix())
    os.chdir(calvin_agent)

    from calvin_agent.evaluation import evaluate_policy as eval_mod
    import hydra
    from omegaconf import OmegaConf

    eval_mod.EP_LEN = args.ep_len
    task_cfg = OmegaConf.load(calvin_models / "conf/callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(calvin_models / "conf/annotations/new_playtable_validation.yaml")

    with open(args.run_meta, encoding="utf-8") as f:
        run_meta = json.load(f)
    task_to_id = {str(task): int(index) for task, index in run_meta["tasks"].items()}
    text_to_task, _ = base.load_eval_annotation_maps(args.calvin_code)
    checkpoint_meta = policy.torch.load(args.checkpoint, map_location="cpu")
    language_mode = str(checkpoint_meta.get("executor_language_mode", "onehot"))
    language_embeddings = (
        policy.load_task_language_embeddings(args.eval_data_root)
        if language_mode == "minilm"
        else None
    )
    model = policy.ChunkFlowPolicy(
        args.checkpoint,
        task_to_id,
        text_to_task,
        device=args.device,
        task_language_embeddings=language_embeddings,
        g_scale=args.g_scale,
        query_g_control=args.query_g_control,
        language_control=args.language_control,
        action_execution_horizon=args.action_execution_horizon,
    )
    if model.executor_state_scope != "visual":
        raise RuntimeError("Source-scene health evaluator is intended for visual checkpoints")

    scene_ranges, archive_root = load_scene_ranges(args.source_archive)
    selected_scenes = sorted(args.scenes or scene_ranges)
    unknown_scenes = set(selected_scenes) - set(scene_ranges)
    if unknown_scenes:
        raise ValueError(f"Unknown source scenes: {sorted(unknown_scenes)}")
    segments = base.load_lang_segments(args.source_data_root / "training")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for segment in segments:
        if segment["task"] in args.tasks:
            scene = scene_for_frame(int(segment["start"]), scene_ranges)
            grouped[(scene, segment["task"])].append(segment)
    for values in grouped.values():
        values.sort(key=lambda item: int(item["start"]))

    selected: list[dict[str, Any]] = []
    for scene in selected_scenes:
        for task in args.tasks:
            for segment in evenly_spaced(grouped[(scene, task)], args.num_per_task_scene):
                selected.append({"scene": scene, "task": task, "start": int(segment["start"])})

    signature = {
        "checkpoint": args.checkpoint.as_posix(),
        "source_data_root": args.source_data_root.as_posix(),
        "source_archive": args.source_archive.as_posix(),
        "eval_data_root": args.eval_data_root.as_posix(),
        "tasks": args.tasks,
        "scenes": selected_scenes,
        "num_per_task_scene": args.num_per_task_scene,
        "ep_len": args.ep_len,
        "g_scale": args.g_scale,
        "query_g_control": args.query_g_control,
        "language_control": args.language_control,
        "action_execution_horizon": args.action_execution_horizon,
        "audit_reset_visuals": args.audit_reset_visuals,
        "scene_ranges": {scene: list(bounds) for scene, bounds in scene_ranges.items()},
        "selected": selected,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.out_dir / "eval_progress.json"
    results: list[dict[str, Any]] = []
    if progress_path.exists():
        with open(progress_path, encoding="utf-8") as f:
            progress = json.load(f)
        for key, expected in signature.items():
            if progress.get(key) != expected:
                raise RuntimeError(f"Progress mismatch for {key}")
        results = list(progress.get("results", []))
        print(f"[resume] {len(results)}/{len(selected)}", flush=True)

    completed = len(results)
    plans: dict[str, list[Any]] = defaultdict(list)
    open_envs: list[Any] = []
    source_frames = zipfile.ZipFile(args.source_archive) if args.audit_reset_visuals else None
    for scene in selected_scenes:
        scene_items = selected[completed:]
        if not any(item["scene"] == scene for item in scene_items):
            continue
        env = make_scene_env(args.calvin_code, args.eval_data_root, scene)
        open_envs.append(env)
        for item in selected[completed:]:
            if item["scene"] != scene:
                continue
            episode_path = args.source_data_root / "training" / f"episode_{item['start']:07d}.npz"
            with np.load(episode_path) as episode:
                robot_obs = np.asarray(episode["robot_obs"], dtype=np.float32)
                scene_obs = np.asarray(episode["scene_obs"], dtype=np.float32)
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            reset_audit: dict[str, float] = {}
            if source_frames is not None:
                member = f"{archive_root}training/episode_{item['start']:07d}.npz"
                with np.load(BytesIO(source_frames.read(member))) as source_episode:
                    stored_rgb = {
                        camera: np.asarray(source_episode[camera], dtype=np.uint8)
                        for camera in ("rgb_static", "rgb_gripper")
                    }
                online_obs = env.get_obs()
                online_features = model._encode_visual_observation(online_obs)
                stored_features = model._encode_visual_observation({"rgb_obs": stored_rgb})
                for camera, online_feature, stored_feature in zip(
                    ("rgb_static", "rgb_gripper"), online_features, stored_features
                ):
                    online_rgb = np.asarray(online_obs["rgb_obs"][camera], dtype=np.float32)
                    reset_audit[f"{camera}_pixel_mae"] = float(
                        np.mean(np.abs(online_rgb - stored_rgb[camera].astype(np.float32)))
                    )
                    denominator = float(np.linalg.norm(online_feature) * np.linalg.norm(stored_feature))
                    reset_audit[f"{camera}_dino_cosine"] = float(
                        np.dot(online_feature, stored_feature) / max(denominator, 1e-12)
                    )
            model.reset()
            success = bool(
                eval_mod.rollout(
                    env,
                    model,
                    task_oracle,
                    item["task"],
                    val_annotations,
                    plans,
                    debug=False,
                )
            )
            results.append({**item, "success": int(success), **reset_audit})
            completed += 1
            save_progress(progress_path, signature, results)
            interval = max(1, len(selected) // 20)
            if completed == 1 or completed == len(selected) or completed % interval == 0:
                print(
                    f"[eval] {completed}/{len(selected)} last={scene}:{item['task']}:{int(success)} "
                    f"sr={sum(row['success'] for row in results) / len(results):.3f}",
                    flush=True,
                )
        # Keep each PyBullet client alive. Explicit close followed by object
        # destruction can disconnect a newly reused client id in CALVIN.

    counts: Counter[tuple[str, str]] = Counter()
    successes: Counter[tuple[str, str]] = Counter()
    for row in results:
        key = (row["scene"], row["task"])
        counts[key] += 1
        successes[key] += int(row["success"])
    summary = {
        **signature,
        "total_success": int(sum(successes.values())),
        "total_rollouts": int(sum(counts.values())),
        "overall_sr": float(sum(successes.values()) / max(1, sum(counts.values()))),
        "cells": {
            f"{scene}/{task}": {
                "success": int(successes[(scene, task)]),
                "total": int(counts[(scene, task)]),
                "sr": float(successes[(scene, task)] / max(1, counts[(scene, task)])),
            }
            for scene in selected_scenes
            for task in args.tasks
        },
    }
    if args.audit_reset_visuals:
        summary["reset_visual_audit"] = {
            metric: {
                "median": float(np.median([row[metric] for row in results])),
                "min": float(np.min([row[metric] for row in results])),
                "max": float(np.max([row[metric] for row in results])),
            }
            for metric in (
                "rgb_static_pixel_mae",
                "rgb_static_dino_cosine",
                "rgb_gripper_pixel_mae",
                "rgb_gripper_dino_cosine",
            )
        }
    with open(args.out_dir / "source_scene_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=base.json_default)
    print(json.dumps(summary, indent=2, default=base.json_default), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
