#!/usr/bin/env python3
"""Diagnose CALVIN slider left/right rollout dynamics for state-BC policies."""

from __future__ import annotations

import argparse
from collections import defaultdict
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


def load_demo_targets(path: Path) -> dict[str, dict[str, np.ndarray]]:
    data = np.load(path, allow_pickle=True)
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in data.files:
        if key.endswith("__starts"):
            task = key[: -len("__starts")]
            target_key = f"{task}__targets"
            if target_key in data:
                out[task] = {
                    "starts": data[key].astype(np.float32),
                    "targets": data[target_key].astype(np.float32),
                }
    return out


def parse_variant(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Variant must be label=/path/to/policy.pt, got {spec}")
    label, path = spec.split("=", 1)
    return label, Path(path).expanduser().resolve()


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.nanmean(arr))


def safe_percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.nanpercentile(arr, q))


def slider_observation(
    robot: np.ndarray,
    scene: np.ndarray,
    geometry: dict[str, Any],
    task: str,
) -> dict[str, Any]:
    joint = geometry["joints"]["slider"]
    axis = np.asarray(joint["axis_unit"], dtype=np.float64)
    intercept = np.asarray(joint["intercept"], dtype=np.float64)
    slope = np.asarray(joint["slope"], dtype=np.float64)
    q = float(scene[int(joint["joint_index"])])
    target = float(pol.articulation_target_controls(task, scene)[0])
    joint_range = float(joint["joint_range"])
    part_point = intercept + slope * q
    eef = np.asarray(robot[:3], dtype=np.float64)
    affordance = part_point - eef
    axis_offset = float(np.dot(affordance, axis))
    lateral = affordance - axis_offset * axis
    residual_norm = (target - q) / max(joint_range, 1e-6)
    contact_threshold = float(joint.get("contact_dist_threshold", 0.04))
    dist = float(np.linalg.norm(affordance))
    if abs(residual_norm) < 0.08:
        stage = "done"
    elif dist > contact_threshold:
        stage = "approach"
    else:
        stage = "actuate"
    return {
        "q": q,
        "target": target,
        "residual_norm": float(residual_norm),
        "dist": dist,
        "axis_offset": axis_offset,
        "lateral_dist": float(np.linalg.norm(lateral)),
        "stage": stage,
        "contact": bool(dist <= contact_threshold),
    }


def summarize_rollout(records: list[dict[str, Any]], task: str, success: bool, success_step: int | None) -> dict[str, Any]:
    if not records:
        return {"task": task, "success": success, "success_step": success_step, "category": "empty"}

    target = records[0]["target"]
    q_start = records[0]["q"]
    denom = max(abs(target - q_start), 1e-6)
    desired_sign = 1.0 if target >= q_start else -1.0
    q_values = [float(r["q"]) for r in records]
    progress_values = [desired_sign * (q - q_start) / denom for q in q_values]
    contact_records = [r for r in records if r["contact"]]
    contact_action_axis = [float(r["action_axis"]) for r in contact_records if r.get("action_axis") is not None]
    contact_axis_offsets = [float(r["axis_offset"]) for r in contact_records]
    contact_q = [float(r["q"]) for r in contact_records]
    first_contact_step = contact_records[0]["step"] if contact_records else None

    final_progress = float(progress_values[-1])
    max_progress = float(max(progress_values))
    min_dist = float(min(float(r["dist"]) for r in records))
    positive_axis_frac = None
    if contact_action_axis:
        positive_axis_frac = float(np.mean(np.asarray(contact_action_axis) * desired_sign > 0.0))

    if success:
        category = "success"
    elif first_contact_step is None:
        category = "never_contact"
    elif max_progress >= 0.80:
        category = "near_success"
    elif contact_action_axis and float(np.mean(contact_action_axis)) * desired_sign <= 0.0:
        category = "contact_wrong_axis_action"
    else:
        category = "contact_insufficient_motion"

    return {
        "task": task,
        "success": bool(success),
        "success_step": success_step,
        "category": category,
        "q_start": float(q_start),
        "q_end": float(q_values[-1]),
        "target": float(target),
        "final_progress": final_progress,
        "max_progress": max_progress,
        "min_dist": min_dist,
        "first_contact_step": first_contact_step,
        "contact_steps": len(contact_records),
        "contact_fraction": float(len(contact_records) / max(1, len(records))),
        "q_at_first_contact": None if not contact_q else float(contact_q[0]),
        "mean_q_contact": safe_mean(contact_q),
        "mean_action_axis_contact": safe_mean(contact_action_axis),
        "p10_action_axis_contact": safe_percentile(contact_action_axis, 10),
        "p90_action_axis_contact": safe_percentile(contact_action_axis, 90),
        "positive_axis_action_fraction_contact": positive_axis_frac,
        "mean_axis_offset_contact": safe_mean(contact_axis_offsets),
        "p10_axis_offset_contact": safe_percentile(contact_axis_offsets, 10),
        "p90_axis_offset_contact": safe_percentile(contact_axis_offsets, 90),
        "num_steps": len(records),
    }


def aggregate_rollouts(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"count": len(rollouts), "success": int(sum(r["success"] for r in rollouts))}
    out["sr"] = float(out["success"] / max(1, out["count"]))
    categories: dict[str, int] = defaultdict(int)
    for r in rollouts:
        categories[str(r["category"])] += 1
    out["categories"] = dict(sorted(categories.items()))
    for key in [
        "final_progress",
        "max_progress",
        "min_dist",
        "first_contact_step",
        "contact_fraction",
        "mean_action_axis_contact",
        "positive_axis_action_fraction_contact",
        "mean_axis_offset_contact",
    ]:
        vals = [float(r[key]) for r in rollouts if r.get(key) is not None]
        out[f"{key}_mean"] = safe_mean(vals)
        out[f"{key}_p50"] = safe_percentile(vals, 50)
    failed = [r for r in rollouts if not r["success"]]
    out["failed_final_progress_p50"] = safe_percentile([r["final_progress"] for r in failed], 50)
    out["failed_max_progress_p50"] = safe_percentile([r["max_progress"] for r in failed], 50)
    out["failed_min_dist_p50"] = safe_percentile([r["min_dist"] for r in failed], 50)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calvin-code", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--demo-targets", type=Path, required=True)
    parser.add_argument("--articulation-geometry-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", required=True, help="label=/abs/path/policy.pt")
    parser.add_argument("--tasks", nargs="+", default=["move_slider_left", "move_slider_right"])
    parser.add_argument("--num-sequences", type=int, default=25)
    parser.add_argument("--scan-sequences", type=int, default=5000)
    parser.add_argument("--ep-len", type=int, default=160)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.calvin_code = args.calvin_code.resolve()
    args.data_root = args.data_root.resolve()
    args.demo_targets = args.demo_targets.resolve()
    args.articulation_geometry_json = args.articulation_geometry_json.resolve()
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
    import hydra
    from omegaconf import OmegaConf

    text_to_task, _task_to_text = pol.load_eval_annotation_maps(args.calvin_code)
    segments = pol.load_lang_segments(args.data_root / "training")
    task_to_id = pol.build_task_index(segments, text_to_task)
    demos = load_demo_targets(args.demo_targets)
    geometry = pol.load_articulation_geometry(args.articulation_geometry_json)
    if geometry is None:
        raise RuntimeError("Missing articulation geometry")
    axis = np.asarray(geometry["joints"]["slider"]["axis_unit"], dtype=np.float64)

    buckets: dict[str, list[Any]] = {task: [] for task in args.tasks}
    for initial_state, seq in get_sequences(max(args.scan_sequences, args.num_sequences * len(args.tasks) * 10)):
        task = seq[0]
        if task in buckets and len(buckets[task]) < args.num_sequences:
            buckets[task].append(initial_state)
        if all(len(items) >= args.num_sequences for items in buckets.values()):
            break
    missing = {task: len(items) for task, items in buckets.items() if len(items) < args.num_sequences}
    if missing:
        raise RuntimeError(f"Could not find enough initial states: {missing}")

    task_cfg = OmegaConf.load(calvin_models / "conf/callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(calvin_models / "conf/annotations/new_playtable_validation.yaml")
    env = eval_mod.make_env(args.data_root)

    report: dict[str, Any] = {
        "tasks": args.tasks,
        "num_sequences": args.num_sequences,
        "ep_len": args.ep_len,
        "slider_axis_unit": axis,
        "variants": {},
    }
    all_rollout_rows: list[dict[str, Any]] = []

    for label, ckpt_path in [parse_variant(spec) for spec in args.variants]:
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        model = pol.StateBCPolicy(ckpt_path, task_to_id, text_to_task, demos, device=args.device)
        variant_rollouts: list[dict[str, Any]] = []
        for task in args.tasks:
            lang_annotation = val_annotations[task][0]
            for rollout_idx, initial_state in enumerate(buckets[task]):
                robot_obs, scene_obs = eval_mod.get_env_state_for_initial_condition(initial_state)
                env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
                obs = env.get_obs()
                start_info = env.get_info()
                model.reset()
                records: list[dict[str, Any]] = []
                success = False
                success_step: int | None = None
                for step in range(args.ep_len):
                    robot = np.asarray(obs["robot_obs"], dtype=np.float32)
                    scene = np.asarray(obs["scene_obs"], dtype=np.float32)
                    state = slider_observation(robot, scene, geometry, task)
                    action = model.step(obs, lang_annotation)
                    state.update(
                        {
                            "step": step,
                            "action_x": float(action[0]),
                            "action_y": float(action[1]),
                            "action_z": float(action[2]),
                            "action_axis": float(np.dot(np.asarray(action[:3], dtype=np.float64), axis)),
                            "action_gripper": float(action[-1]),
                        }
                    )
                    obs, _, _, current_info = env.step(action)
                    current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {task})
                    if len(current_task_info) > 0 and not success:
                        success = True
                        success_step = step
                    records.append(state)
                    if success and success_step is not None and step >= success_step + 5:
                        break

                summary = summarize_rollout(records, task=task, success=success, success_step=success_step)
                summary.update({"variant": label, "checkpoint": ckpt_path, "rollout_idx": rollout_idx})
                variant_rollouts.append(summary)
                all_rollout_rows.append(summary)
                print(
                    f"[diag] {label} {task} {rollout_idx:02d} "
                    f"success={int(success)} cat={summary['category']} "
                    f"final_prog={summary.get('final_progress'):.3f} min_dist={summary.get('min_dist'):.3f}",
                    flush=True,
                )

        by_task = {}
        for task in args.tasks:
            by_task[task] = aggregate_rollouts([r for r in variant_rollouts if r["task"] == task])
        report["variants"][label] = {"checkpoint": ckpt_path, "by_task": by_task, "rollouts": variant_rollouts}

    with open(args.out_dir / "left_right_rollout_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=json_default)
    with open(args.out_dir / "left_right_rollout_rows.json", "w", encoding="utf-8") as f:
        json.dump(all_rollout_rows, f, indent=2, default=json_default)

    compact = {
        label: {
            task: {
                "sr": stats["sr"],
                "categories": stats["categories"],
                "final_progress_p50": stats["final_progress_p50"],
                "failed_max_progress_p50": stats["failed_max_progress_p50"],
                "failed_min_dist_p50": stats["failed_min_dist_p50"],
                "mean_action_axis_contact_p50": stats["mean_action_axis_contact_p50"],
                "mean_axis_offset_contact_p50": stats["mean_axis_offset_contact_p50"],
            }
            for task, stats in variant["by_task"].items()
        }
        for label, variant in report["variants"].items()
    }
    with open(args.out_dir / "compact_summary.json", "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2, default=json_default)
    print(json.dumps(compact, indent=2, default=json_default), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
