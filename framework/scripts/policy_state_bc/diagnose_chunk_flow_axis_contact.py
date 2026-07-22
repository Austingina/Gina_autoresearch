#!/usr/bin/env python3
"""Diagnose whether chunk policies satisfy articulation axes in non-useful poses."""

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
        raise ValueError(f"Policy spec must be alias=/path/to/policy.pt, got {spec!r}")
    alias, path = spec.split("=", 1)
    alias = alias.strip()
    if not alias:
        raise ValueError(f"Empty alias in policy spec: {spec!r}")
    return alias, Path(path).expanduser()


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.nanmean(np.asarray(values, dtype=np.float64)))


def safe_percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.nanpercentile(np.asarray(values, dtype=np.float64), q))


def safe_sum(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.nansum(np.asarray(values, dtype=np.float64)))


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def axis_perp_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
    ref = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(axis, ref))) > 0.9:
        ref = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    b1 = np.cross(axis, ref)
    b1 = b1 / max(float(np.linalg.norm(b1)), 1e-8)
    b2 = np.cross(axis, b1)
    b2 = b2 / max(float(np.linalg.norm(b2)), 1e-8)
    return b1, b2


def articulation_frame(
    robot: np.ndarray,
    scene: np.ndarray,
    geometry: dict[str, Any],
    task: str,
) -> dict[str, Any]:
    joint_name = base.ARTICULATION_TASK_TO_JOINT[task]
    joint = geometry["joints"][joint_name]
    joint_idx = int(joint["joint_index"])
    axis = np.asarray(joint["axis_unit"], dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
    b1, b2 = axis_perp_basis(axis)
    intercept = np.asarray(joint["intercept"], dtype=np.float64)
    slope = np.asarray(joint["slope"], dtype=np.float64)
    q = float(scene[joint_idx])
    target = float(base.articulation_target_controls(task, scene)[joint_idx])
    joint_range = float(joint["joint_range"])
    residual = target - q
    desired_sign = 1.0 if residual >= 0.0 else -1.0
    desired_axis = axis * desired_sign
    part_point = intercept + slope * q
    eef = np.asarray(robot[:3], dtype=np.float64)
    rel_eef_from_part = eef - part_point
    part_to_eef_axis = float(np.dot(rel_eef_from_part, desired_axis))
    part_to_eef_b1 = float(np.dot(rel_eef_from_part, b1))
    part_to_eef_b2 = float(np.dot(rel_eef_from_part, b2))
    line_lateral = rel_eef_from_part - np.dot(rel_eef_from_part, axis) * axis
    dist_to_part = float(np.linalg.norm(rel_eef_from_part))
    dist_to_axis_line = float(np.linalg.norm(line_lateral))
    threshold = float(joint.get("contact_dist_threshold", 0.05))
    radius = float(max(0.02, min(0.08, threshold)))
    residual_mag = abs(float(residual / max(joint_range, 1e-6)))
    contact_anchor = part_point - desired_axis * radius
    eef_to_contact = contact_anchor - eef
    eef_to_contact_norm = float(np.linalg.norm(eef_to_contact))
    if eef_to_contact_norm < 1e-8:
        g_eef_field = np.zeros(3, dtype=np.float64)
    else:
        g_eef_field = (
            eef_to_contact
            / eef_to_contact_norm
            * min(1.25, eef_to_contact_norm / max(radius, 1e-6))
        )
    g_desired_flow = desired_axis * min(residual_mag, 1.25)
    return {
        "joint_name": joint_name,
        "joint_idx": joint_idx,
        "q": q,
        "target": target,
        "joint_range": joint_range,
        "desired_sign": desired_sign,
        "residual_norm": float(residual / max(joint_range, 1e-6)),
        "axis_x": float(axis[0]),
        "axis_y": float(axis[1]),
        "axis_z": float(axis[2]),
        "desired_axis_x": float(desired_axis[0]),
        "desired_axis_y": float(desired_axis[1]),
        "desired_axis_z": float(desired_axis[2]),
        "eef_x": float(eef[0]),
        "eef_y": float(eef[1]),
        "eef_z": float(eef[2]),
        "part_x": float(part_point[0]),
        "part_y": float(part_point[1]),
        "part_z": float(part_point[2]),
        "part_to_eef_axis": part_to_eef_axis,
        "part_to_eef_b1": part_to_eef_b1,
        "part_to_eef_b2": part_to_eef_b2,
        "eef_height_minus_part": float(eef[2] - part_point[2]),
        "dist_to_part": dist_to_part,
        "dist_to_axis_line": dist_to_axis_line,
        "contact_anchor_x": float(contact_anchor[0]),
        "contact_anchor_y": float(contact_anchor[1]),
        "contact_anchor_z": float(contact_anchor[2]),
        "dist_to_contact_anchor": eef_to_contact_norm,
        "near_contact_anchor": bool(eef_to_contact_norm <= threshold),
        "g_eef_field_x": float(g_eef_field[0]),
        "g_eef_field_y": float(g_eef_field[1]),
        "g_eef_field_z": float(g_eef_field[2]),
        "g_eef_field_norm": float(np.linalg.norm(g_eef_field)),
        "g_eef_field_desired_axis": float(np.dot(g_eef_field, desired_axis)),
        "g_desired_flow_x": float(g_desired_flow[0]),
        "g_desired_flow_y": float(g_desired_flow[1]),
        "g_desired_flow_z": float(g_desired_flow[2]),
        "g_desired_flow_norm": float(np.linalg.norm(g_desired_flow)),
        "g_desired_flow_desired_axis": float(np.dot(g_desired_flow, desired_axis)),
        "near_part": bool(dist_to_part <= threshold),
        "near_axis_line": bool(dist_to_axis_line <= threshold),
        "contact_threshold": threshold,
    }


def summarize_records(records: list[dict[str, Any]], success: bool, success_step: int | None) -> dict[str, Any]:
    if not records:
        return {"success": bool(success), "success_step": success_step, "category": "empty"}
    q0 = float(records[0]["q"])
    target = float(records[0]["target"])
    desired_sign = float(records[0]["desired_sign"])
    denom = max(abs(target - q0), 1e-6)
    q_progress = [desired_sign * (float(r["q"]) - q0) / denom for r in records]
    action_axis = [float(r["action_desired_axis"]) for r in records]
    prior_axis = [float(r["prior_action_desired_axis"]) for r in records if r.get("prior_action_desired_axis") is not None]
    learned_axis = [
        float(r["learned_action_desired_axis"]) for r in records if r.get("learned_action_desired_axis") is not None
    ]
    eef_axis_motion = [float(r["eef_delta_desired_axis"]) for r in records]
    q_delta = [float(r["q_delta_desired"]) for r in records]
    near_axis = [r for r in records if r["near_axis_line"]]
    near_part = [r for r in records if r["near_part"]]
    near_anchor = [r for r in records if r["near_contact_anchor"]]
    axis_cmd = [r for r in records if r["axis_command"]]
    weird = [r for r in records if r["axis_command_no_progress"] and r["near_axis_line"] and not r["near_part"]]
    wrong_side = [r for r in records if r["axis_command"] and float(r["part_to_eef_axis"]) > 0.0]
    useful_push = [r for r in records if r["axis_command"] and r["near_part"] and float(r["q_delta_desired"]) > 1e-5]
    first_near_part_step = min((int(r["step"]) for r in near_part), default=None)
    first_near_axis_line_step = min((int(r["step"]) for r in near_axis), default=None)
    first_near_anchor_step = min((int(r["step"]) for r in near_anchor), default=None)
    first_joint_progress_step = min(
        (int(r["step"]) for r in records if float(r["q_delta_desired"]) > 1e-5),
        default=None,
    )
    if first_near_part_step is None:
        pre_contact = records
        post_contact: list[dict[str, Any]] = []
        q_progress_at_first_near_part = None
        q_progress_after_first_near_part = None
    else:
        pre_contact = [r for r in records if int(r["step"]) < first_near_part_step]
        post_contact = [r for r in records if int(r["step"]) >= first_near_part_step]
        q_progress_at_first_near_part = float(q_progress[first_near_part_step])
        q_progress_after_first_near_part = float(q_progress[-1] - q_progress_at_first_near_part)

    if success:
        category = "success"
    elif weird:
        category = "axis_aligned_but_not_contacting"
    elif near_part and not useful_push:
        category = "near_part_no_joint_progress"
    elif near_axis and not near_part:
        category = "near_axis_line_wrong_offset"
    else:
        category = "no_axis_contact"

    def values(rows: list[dict[str, Any]], key: str) -> list[float]:
        return [float(r[key]) for r in rows if r.get(key) is not None]

    return {
        "success": bool(success),
        "success_step": success_step,
        "category": category,
        "num_steps": len(records),
        "final_progress": float(q_progress[-1]),
        "max_progress": float(max(q_progress)),
        "min_dist_to_part": float(min(float(r["dist_to_part"]) for r in records)),
        "min_dist_to_axis_line": float(min(float(r["dist_to_axis_line"]) for r in records)),
        "initial_dist_to_part": float(records[0]["dist_to_part"]),
        "final_dist_to_part": float(records[-1]["dist_to_part"]),
        "initial_dist_to_contact_anchor": float(records[0]["dist_to_contact_anchor"]),
        "min_dist_to_contact_anchor": float(min(float(r["dist_to_contact_anchor"]) for r in records)),
        "final_dist_to_contact_anchor": float(records[-1]["dist_to_contact_anchor"]),
        "first_near_part_step": first_near_part_step,
        "first_near_axis_line_step": first_near_axis_line_step,
        "first_near_contact_anchor_step": first_near_anchor_step,
        "first_joint_progress_step": first_joint_progress_step,
        "q_progress_at_first_near_part": q_progress_at_first_near_part,
        "q_progress_after_first_near_part": q_progress_after_first_near_part,
        "near_part_steps": len(near_part),
        "near_axis_line_steps": len(near_axis),
        "near_contact_anchor_steps": len(near_anchor),
        "axis_command_steps": len(axis_cmd),
        "axis_command_no_progress_steps": int(sum(1 for r in records if r["axis_command_no_progress"])),
        "weird_axis_satisfied_steps": len(weird),
        "wrong_side_axis_command_steps": len(wrong_side),
        "useful_push_steps": len(useful_push),
        "mean_action_desired_axis": safe_mean(action_axis),
        "mean_prior_action_desired_axis": safe_mean(prior_axis),
        "mean_learned_action_desired_axis": safe_mean(learned_axis),
        "mean_eef_delta_desired_axis": safe_mean(eef_axis_motion),
        "mean_q_delta_desired": safe_mean(q_delta),
        "pre_contact_mean_action_desired_axis": safe_mean(values(pre_contact, "action_desired_axis")),
        "pre_contact_mean_prior_action_desired_axis": safe_mean(
            values(pre_contact, "prior_action_desired_axis")
        ),
        "pre_contact_mean_learned_action_desired_axis": safe_mean(
            values(pre_contact, "learned_action_desired_axis")
        ),
        "post_contact_mean_action_desired_axis": safe_mean(values(post_contact, "action_desired_axis")),
        "pre_contact_mean_eef_delta_desired_axis": safe_mean(values(pre_contact, "eef_delta_desired_axis")),
        "post_contact_mean_eef_delta_desired_axis": safe_mean(values(post_contact, "eef_delta_desired_axis")),
        "pre_contact_sum_q_delta_desired": safe_sum(values(pre_contact, "q_delta_desired")),
        "post_contact_sum_q_delta_desired": safe_sum(values(post_contact, "q_delta_desired")),
        "post_contact_mean_q_delta_desired": safe_mean(values(post_contact, "q_delta_desired")),
        "pre_contact_mean_action_norm": safe_mean(values(pre_contact, "action_norm")),
        "post_contact_mean_action_norm": safe_mean(values(post_contact, "action_norm")),
        "pre_contact_mean_eef_step_norm": safe_mean(values(pre_contact, "eef_step_norm")),
        "post_contact_mean_eef_step_norm": safe_mean(values(post_contact, "eef_step_norm")),
        "pre_contact_mean_action_g_field_cos": safe_mean(values(pre_contact, "action_g_eef_field_cos")),
        "pre_contact_mean_eef_g_field_cos": safe_mean(values(pre_contact, "eef_delta_g_eef_field_cos")),
        "pre_contact_mean_field_gain": safe_mean(values(pre_contact, "structured_field_gain")),
        "pre_contact_mean_orthogonal_scale": safe_mean(values(pre_contact, "structured_orthogonal_scale")),
        "pre_contact_command_clipped_fraction": safe_mean(
            [float(r["command_was_clipped"]) for r in pre_contact]
        ),
        "pre_contact_mean_unclipped_action_norm": safe_mean(values(pre_contact, "unclipped_action_norm")),
        "near_part_mean_action_norm": safe_mean(values(near_part, "action_norm")),
        "near_part_mean_action_desired_axis": safe_mean(values(near_part, "action_desired_axis")),
        "near_part_mean_eef_delta_desired_axis": safe_mean(values(near_part, "eef_delta_desired_axis")),
        "eef_net_desired_axis": float(
            np.dot(
                np.asarray([records[-1]["eef_x"], records[-1]["eef_y"], records[-1]["eef_z"]])
                - np.asarray([records[0]["eef_x"], records[0]["eef_y"], records[0]["eef_z"]]),
                np.asarray(
                    [records[0]["desired_axis_x"], records[0]["desired_axis_y"], records[0]["desired_axis_z"]]
                ),
            )
        ),
        "near_axis_mean_part_dist": safe_mean(values(near_axis, "dist_to_part")),
        "near_axis_mean_line_dist": safe_mean(values(near_axis, "dist_to_axis_line")),
        "near_axis_mean_part_to_eef_axis": safe_mean(values(near_axis, "part_to_eef_axis")),
        "near_axis_mean_height_offset": safe_mean(values(near_axis, "eef_height_minus_part")),
        "weird_mean_part_to_eef_axis": safe_mean(values(weird, "part_to_eef_axis")),
        "weird_mean_height_offset": safe_mean(values(weird, "eef_height_minus_part")),
        "weird_mean_action_desired_axis": safe_mean(values(weird, "action_desired_axis")),
        "weird_mean_q_delta_desired": safe_mean(values(weird, "q_delta_desired")),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"count": len(rows), "success": int(sum(bool(r["success"]) for r in rows))}
    out["sr"] = float(out["success"] / max(1, len(rows)))
    cats: dict[str, int] = defaultdict(int)
    for row in rows:
        cats[str(row["category"])] += 1
    out["categories"] = dict(sorted(cats.items()))
    for key in [
        "final_progress",
        "max_progress",
        "min_dist_to_part",
        "min_dist_to_axis_line",
        "initial_dist_to_part",
        "final_dist_to_part",
        "initial_dist_to_contact_anchor",
        "min_dist_to_contact_anchor",
        "final_dist_to_contact_anchor",
        "first_near_part_step",
        "first_near_axis_line_step",
        "first_near_contact_anchor_step",
        "first_joint_progress_step",
        "q_progress_at_first_near_part",
        "q_progress_after_first_near_part",
        "near_part_steps",
        "near_axis_line_steps",
        "near_contact_anchor_steps",
        "axis_command_steps",
        "axis_command_no_progress_steps",
        "weird_axis_satisfied_steps",
        "wrong_side_axis_command_steps",
        "useful_push_steps",
        "pre_contact_mean_action_desired_axis",
        "pre_contact_mean_prior_action_desired_axis",
        "pre_contact_mean_learned_action_desired_axis",
        "post_contact_mean_action_desired_axis",
        "pre_contact_mean_eef_delta_desired_axis",
        "post_contact_mean_eef_delta_desired_axis",
        "pre_contact_sum_q_delta_desired",
        "post_contact_sum_q_delta_desired",
        "post_contact_mean_q_delta_desired",
        "pre_contact_mean_action_norm",
        "post_contact_mean_action_norm",
        "pre_contact_mean_eef_step_norm",
        "post_contact_mean_eef_step_norm",
        "pre_contact_mean_action_g_field_cos",
        "pre_contact_mean_eef_g_field_cos",
        "pre_contact_mean_field_gain",
        "pre_contact_mean_orthogonal_scale",
        "pre_contact_command_clipped_fraction",
        "pre_contact_mean_unclipped_action_norm",
        "near_part_mean_action_norm",
        "near_part_mean_action_desired_axis",
        "near_part_mean_eef_delta_desired_axis",
        "eef_net_desired_axis",
        "near_axis_mean_part_to_eef_axis",
        "near_axis_mean_height_offset",
        "weird_mean_part_to_eef_axis",
        "weird_mean_height_offset",
        "weird_mean_action_desired_axis",
        "weird_mean_q_delta_desired",
    ]:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        out[f"{key}_mean"] = safe_mean(vals)
        out[f"{key}_p50"] = safe_percentile(vals, 50)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calvin-code", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--articulation-geometry-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", required=True, help="alias=/path/to/policy.pt entries")
    parser.add_argument("--tasks", nargs="+", default=["close_drawer"])
    parser.add_argument("--num-sequences", type=int, default=25)
    parser.add_argument("--rollout-indices", nargs="*", type=int, default=None)
    parser.add_argument("--scan-sequences", type=int, default=5000)
    parser.add_argument("--ep-len", type=int, default=160)
    parser.add_argument("--axis-action-threshold", type=float, default=0.02)
    parser.add_argument("--q-delta-threshold", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--g-scale", type=float, default=1.0)
    parser.add_argument("--query-g-control", default="normal")
    parser.add_argument("--language-control", default="normal")
    parser.add_argument("--action-execution-horizon", type=int, default=1)
    parser.add_argument("--policy-sample-seed", type=int, default=None)
    parser.add_argument("--fm-reseed-each-episode", type=int, choices=[0, 1], default=1)
    parser.add_argument("--state-only-observation", action="store_true")
    args = parser.parse_args()

    launch_cwd = Path.cwd().resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else (launch_cwd / path).resolve()

    args.calvin_code = resolve(args.calvin_code)
    args.data_root = resolve(args.data_root)
    args.articulation_geometry_json = resolve(args.articulation_geometry_json)
    args.out_dir = resolve(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    policies = [(alias, resolve(path)) for alias, path in map(parse_policy_spec, args.policies)]

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
    geometry = base.load_articulation_geometry(args.articulation_geometry_json)
    if geometry is None:
        raise RuntimeError("Missing articulation geometry")

    selected_indices = sorted(set(args.rollout_indices or range(args.num_sequences)))
    candidate_count = max(selected_indices) + 1
    buckets: dict[str, list[Any]] = {task: [] for task in args.tasks}
    for initial_state, seq in get_sequences(max(args.scan_sequences, candidate_count * len(args.tasks) * 10)):
        first_task = seq[0]
        if first_task in buckets and len(buckets[first_task]) < candidate_count:
            buckets[first_task].append(initial_state)
        if all(len(items) >= candidate_count for items in buckets.values()):
            break
    missing = {task: len(items) for task, items in buckets.items() if len(items) < candidate_count}
    if missing:
        raise RuntimeError(f"Could not find enough initial states: {missing}")

    task_cfg = OmegaConf.load(calvin_models / "conf/callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(calvin_models / "conf/annotations/new_playtable_validation.yaml")
    env = eval_mod.make_env(args.data_root)
    if args.state_only_observation:
        env.get_obs = env.get_state_obs

    report: dict[str, Any] = {
        "tasks": args.tasks,
        "num_sequences": args.num_sequences,
        "rollout_indices": selected_indices,
        "ep_len": args.ep_len,
        "axis_action_threshold": args.axis_action_threshold,
        "q_delta_threshold": args.q_delta_threshold,
        "g_scale": args.g_scale,
        "query_g_control": args.query_g_control,
        "language_control": args.language_control,
        "policies": {},
    }
    all_rollouts: list[dict[str, Any]] = []
    all_steps: list[dict[str, Any]] = []

    for alias, ckpt_path in policies:
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        model = chunk_pol.ChunkFlowPolicy(
            ckpt_path,
            task_to_id,
            text_to_task,
            device=args.device,
            g_scale=args.g_scale,
            query_g_control=args.query_g_control,
            language_control=args.language_control,
            action_execution_horizon=args.action_execution_horizon,
            sample_seed_override=args.policy_sample_seed,
            reseed_each_episode=bool(args.fm_reseed_each_episode),
        )
        policy_rollouts: list[dict[str, Any]] = []
        for task in args.tasks:
            lang_annotation = val_annotations[task][0]
            for rollout_idx in selected_indices:
                initial_state = buckets[task][rollout_idx]
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
                    frame = articulation_frame(robot, scene, geometry, task)
                    desired_axis = np.asarray(
                        [frame["desired_axis_x"], frame["desired_axis_y"], frame["desired_axis_z"]],
                        dtype=np.float64,
                    )
                    q_before = float(frame["q"])
                    eef_before = np.asarray([frame["eef_x"], frame["eef_y"], frame["eef_z"]], dtype=np.float64)
                    action = model.step(obs, lang_annotation)
                    action_command = np.asarray(action, dtype=np.float64).copy()
                    policy_debug = getattr(model, "last_debug", {})
                    prior_action = np.asarray(policy_debug.get("prior_action", np.zeros(7)), dtype=np.float64)
                    learned_action = np.asarray(policy_debug.get("learned_action", np.zeros(7)), dtype=np.float64)
                    unclipped_action = np.asarray(policy_debug.get("unclipped_action", action), dtype=np.float64)
                    has_components = bool(policy_debug)
                    obs, _, _, current_info = env.step(action)
                    next_robot = np.asarray(obs["robot_obs"], dtype=np.float32)
                    next_scene = np.asarray(obs["scene_obs"], dtype=np.float32)
                    next_frame = articulation_frame(next_robot, next_scene, geometry, task)
                    q_after = float(next_frame["q"])
                    eef_after = np.asarray([next_frame["eef_x"], next_frame["eef_y"], next_frame["eef_z"]], dtype=np.float64)
                    action_desired_axis = float(np.dot(action_command[:3], desired_axis))
                    eef_delta_desired_axis = float(np.dot(eef_after - eef_before, desired_axis))
                    eef_delta = eef_after - eef_before
                    g_eef_field = np.asarray(
                        [frame["g_eef_field_x"], frame["g_eef_field_y"], frame["g_eef_field_z"]],
                        dtype=np.float64,
                    )
                    action_xyz = action_command[:3]
                    q_delta_desired = float(float(frame["desired_sign"]) * (q_after - q_before))
                    axis_command = action_desired_axis > args.axis_action_threshold
                    no_progress = q_delta_desired <= args.q_delta_threshold
                    frame.update(
                        {
                            "policy": alias,
                            "checkpoint": ckpt_path,
                            "task": task,
                            "rollout_idx": rollout_idx,
                            "step": step,
                            "action_x": float(action_command[0]),
                            "action_y": float(action_command[1]),
                            "action_z": float(action_command[2]),
                            "action_gripper": float(action_command[-1]),
                            "action_norm": float(np.linalg.norm(action_xyz)),
                            "action_desired_axis": action_desired_axis,
                            "prior_action_desired_axis": (
                                float(np.dot(prior_action[:3], desired_axis)) if has_components else None
                            ),
                            "learned_action_desired_axis": (
                                float(np.dot(learned_action[:3], desired_axis)) if has_components else None
                            ),
                            "unclipped_action_desired_axis": (
                                float(np.dot(unclipped_action[:3], desired_axis)) if has_components else None
                            ),
                            "unclipped_action_norm": float(np.linalg.norm(unclipped_action[:3])),
                            "unclipped_action_max_abs": float(np.max(np.abs(unclipped_action[:3]))),
                            "command_was_clipped": bool(np.any(np.abs(unclipped_action[:3]) > 1.0 + 1e-6)),
                            "structured_readiness": policy_debug.get("readiness"),
                            "structured_approach_gain": policy_debug.get("approach_gain"),
                            "structured_flow_gain": policy_debug.get("flow_gain"),
                            "structured_field_gain": policy_debug.get("field_gain"),
                            "structured_orthogonal_scale": policy_debug.get("orthogonal_scale"),
                            "eef_delta_desired_axis": eef_delta_desired_axis,
                            "eef_step_norm": float(np.linalg.norm(eef_delta)),
                            "action_g_eef_field_cos": safe_cosine(action_xyz, g_eef_field),
                            "eef_delta_g_eef_field_cos": safe_cosine(eef_delta, g_eef_field),
                            "q_after": q_after,
                            "q_delta_desired": q_delta_desired,
                            "axis_command": bool(axis_command),
                            "axis_command_no_progress": bool(axis_command and no_progress),
                        }
                    )
                    current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {task})
                    if len(current_task_info) > 0 and not success:
                        success = True
                        success_step = step
                    records.append(frame)
                    all_steps.append(frame)
                    if success and success_step is not None and step >= success_step + 5:
                        break
                summary = summarize_records(records, success=success, success_step=success_step)
                summary.update({"policy": alias, "checkpoint": ckpt_path, "task": task, "rollout_idx": rollout_idx})
                policy_rollouts.append(summary)
                all_rollouts.append(summary)
                print(
                    f"[axisdiag] {alias} {task} {rollout_idx:02d} "
                    f"success={int(success)} cat={summary['category']} "
                    f"weird={summary['weird_axis_satisfied_steps']} "
                    f"near_axis={summary['near_axis_line_steps']} near_part={summary['near_part_steps']} "
                    f"max_prog={summary['max_progress']:.3f}",
                    flush=True,
                )
        by_task = {task: aggregate([r for r in policy_rollouts if r["task"] == task]) for task in args.tasks}
        report["policies"][alias] = {"checkpoint": ckpt_path, "by_task": by_task, "rollouts": policy_rollouts}

    with open(args.out_dir / "axis_contact_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=json_default)
    with open(args.out_dir / "axis_contact_rollouts.json", "w", encoding="utf-8") as f:
        json.dump(all_rollouts, f, indent=2, default=json_default)
    with open(args.out_dir / "axis_contact_steps.json", "w", encoding="utf-8") as f:
        json.dump(all_steps, f, indent=2, default=json_default)
    compact = {
        alias: {
            task: {
                "sr": stats["sr"],
                "success": stats["success"],
                "count": stats["count"],
                "categories": stats["categories"],
                "near_axis_line_steps_p50": stats["near_axis_line_steps_p50"],
                "near_part_steps_p50": stats["near_part_steps_p50"],
                "axis_command_no_progress_steps_p50": stats["axis_command_no_progress_steps_p50"],
                "weird_axis_satisfied_steps_p50": stats["weird_axis_satisfied_steps_p50"],
                "wrong_side_axis_command_steps_p50": stats["wrong_side_axis_command_steps_p50"],
                "useful_push_steps_p50": stats["useful_push_steps_p50"],
                "min_dist_to_part_p50": stats["min_dist_to_part_p50"],
                "min_dist_to_axis_line_p50": stats["min_dist_to_axis_line_p50"],
                "min_dist_to_contact_anchor_p50": stats["min_dist_to_contact_anchor_p50"],
                "final_dist_to_part_p50": stats["final_dist_to_part_p50"],
                "first_near_part_step_p50": stats["first_near_part_step_p50"],
                "first_near_axis_line_step_p50": stats["first_near_axis_line_step_p50"],
                "first_near_contact_anchor_step_p50": stats["first_near_contact_anchor_step_p50"],
                "first_joint_progress_step_p50": stats["first_joint_progress_step_p50"],
                "q_progress_after_first_near_part_p50": stats["q_progress_after_first_near_part_p50"],
                "pre_contact_mean_action_desired_axis_p50": stats["pre_contact_mean_action_desired_axis_p50"],
                "pre_contact_mean_prior_action_desired_axis_p50": stats[
                    "pre_contact_mean_prior_action_desired_axis_p50"
                ],
                "pre_contact_mean_learned_action_desired_axis_p50": stats[
                    "pre_contact_mean_learned_action_desired_axis_p50"
                ],
                "post_contact_mean_action_desired_axis_p50": stats["post_contact_mean_action_desired_axis_p50"],
                "pre_contact_mean_eef_delta_desired_axis_p50": stats["pre_contact_mean_eef_delta_desired_axis_p50"],
                "post_contact_mean_eef_delta_desired_axis_p50": stats["post_contact_mean_eef_delta_desired_axis_p50"],
                "pre_contact_sum_q_delta_desired_p50": stats["pre_contact_sum_q_delta_desired_p50"],
                "post_contact_sum_q_delta_desired_p50": stats["post_contact_sum_q_delta_desired_p50"],
                "post_contact_mean_q_delta_desired_p50": stats["post_contact_mean_q_delta_desired_p50"],
                "pre_contact_mean_action_norm_p50": stats["pre_contact_mean_action_norm_p50"],
                "post_contact_mean_action_norm_p50": stats["post_contact_mean_action_norm_p50"],
                "pre_contact_mean_eef_step_norm_p50": stats["pre_contact_mean_eef_step_norm_p50"],
                "post_contact_mean_eef_step_norm_p50": stats["post_contact_mean_eef_step_norm_p50"],
                "pre_contact_mean_action_g_field_cos_p50": stats["pre_contact_mean_action_g_field_cos_p50"],
                "pre_contact_mean_eef_g_field_cos_p50": stats["pre_contact_mean_eef_g_field_cos_p50"],
                "pre_contact_mean_field_gain_p50": stats["pre_contact_mean_field_gain_p50"],
                "pre_contact_mean_orthogonal_scale_p50": stats[
                    "pre_contact_mean_orthogonal_scale_p50"
                ],
                "pre_contact_command_clipped_fraction_p50": stats[
                    "pre_contact_command_clipped_fraction_p50"
                ],
                "pre_contact_mean_unclipped_action_norm_p50": stats[
                    "pre_contact_mean_unclipped_action_norm_p50"
                ],
                "near_part_mean_action_norm_p50": stats["near_part_mean_action_norm_p50"],
                "near_part_mean_action_desired_axis_p50": stats["near_part_mean_action_desired_axis_p50"],
                "near_part_mean_eef_delta_desired_axis_p50": stats[
                    "near_part_mean_eef_delta_desired_axis_p50"
                ],
                "eef_net_desired_axis_p50": stats["eef_net_desired_axis_p50"],
                "near_axis_mean_part_to_eef_axis_p50": stats["near_axis_mean_part_to_eef_axis_p50"],
                "near_axis_mean_height_offset_p50": stats["near_axis_mean_height_offset_p50"],
                "weird_mean_action_desired_axis_p50": stats["weird_mean_action_desired_axis_p50"],
                "weird_mean_q_delta_desired_p50": stats["weird_mean_q_delta_desired_p50"],
            }
            for task, stats in policy["by_task"].items()
        }
        for alias, policy in report["policies"].items()
    }
    with open(args.out_dir / "compact_summary.json", "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2, default=json_default)
    print(json.dumps(compact, indent=2, default=json_default), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
