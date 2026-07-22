#!/usr/bin/env python3
"""Audit whether minimal-G token policies change actions when only G changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, SCRIPT_DIR.as_posix())

import train_eval_state_bc_task_policy as base  # noqa: E402
import train_eval_state_chunk_flow_policy as policy  # noqa: E402


def stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean": None, "p50": None, "p90": None}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    output = np.full(len(a), np.nan, dtype=np.float64)
    valid = denominator > 1e-8
    output[valid] = np.sum(a[valid] * b[valid], axis=1) / denominator[valid]
    return output


class BatchPredictor:
    def __init__(self, checkpoint: Path, device: str, batch_size: int):
        self.checkpoint_path = checkpoint
        self.checkpoint = torch.load(checkpoint, map_location="cpu")
        self.mode = str(self.checkpoint["mode"])
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.base_dim = int(self.checkpoint["base_dim"])
        self.g_dim = int(self.checkpoint["g_dim"])
        self.chunk_size = int(self.checkpoint["chunk_size"])
        if self.mode == "chunk_bc_minimal_g_tokens":
            self.model: torch.nn.Module = policy.MinimalGTokenChunkTransformer(
                self.base_dim,
                self.g_dim,
                self.chunk_size,
                int(self.checkpoint["hidden"]),
                int(self.checkpoint["layers"]),
            )
        elif self.mode == "chunk_fm_minimal_g_tokens":
            self.model = policy.MinimalGTokenFlowTransformer(
                self.base_dim,
                self.g_dim,
                self.chunk_size,
                int(self.checkpoint["hidden"]),
                int(self.checkpoint["layers"]),
            )
        else:
            raise ValueError(f"Unsupported checkpoint mode: {self.mode}")
        self.model.load_state_dict(self.checkpoint["model_state"])
        self.model.to(self.device).eval()
        self.cond_mean = np.asarray(self.checkpoint["cond_mean"], dtype=np.float32)
        self.cond_std = np.asarray(self.checkpoint["cond_std"], dtype=np.float32)
        self.y_mean = np.asarray(self.checkpoint["y_mean"], dtype=np.float32)
        self.y_std = np.asarray(self.checkpoint["y_std"], dtype=np.float32)
        self.fm_steps = int(self.checkpoint.get("fm_steps", 10))
        self.sample_seed = int(self.checkpoint.get("sample_seed", 17))

    def predict(self, base_raw: np.ndarray, g_raw: np.ndarray) -> np.ndarray:
        output: list[np.ndarray] = []
        for start in range(0, len(base_raw), self.batch_size):
            end = min(start + self.batch_size, len(base_raw))
            base_n = (
                base_raw[start:end] - self.cond_mean[None, : self.base_dim]
            ) / self.cond_std[None, : self.base_dim]
            base_t = torch.from_numpy(base_n.astype(np.float32)).to(self.device)
            g_t = torch.from_numpy(g_raw[start:end].astype(np.float32)).to(self.device)
            with torch.no_grad():
                if self.mode == "chunk_bc_minimal_g_tokens":
                    prediction_n = self.model(base_t, g_t)
                else:
                    generator = torch.Generator(device=self.device)
                    generator.manual_seed(self.sample_seed + start)
                    prediction_n = policy.sample_minimal_g_token_flow(
                        self.model,
                        base_t,
                        g_t,
                        steps=self.fm_steps,
                        generator=generator,
                    )
            prediction = prediction_n.cpu().numpy() * self.y_std + self.y_mean
            output.append(prediction.reshape(-1, self.chunk_size, 7))
        return np.concatenate(output, axis=0)


def model_report(
    predictor: BatchPredictor,
    base_x: np.ndarray,
    g_x: np.ndarray,
    target: np.ndarray,
    task_names: np.ndarray,
    inverse_pairs: dict[str, np.ndarray],
) -> dict[str, Any]:
    zero_g = np.zeros_like(g_x)
    flip_g = g_x.copy()
    flip_g[:, 3:6] *= -1.0
    normal = predictor.predict(base_x, g_x)
    zero = predictor.predict(base_x, zero_g)
    flip = predictor.predict(base_x, flip_g)

    per_task: dict[str, Any] = {}
    for task in sorted(set(task_names.astype(str))):
        selected = task_names.astype(str) == task
        selected_flow = g_x[selected, 3:6]
        selected_flow_norm = np.linalg.norm(selected_flow, axis=1, keepdims=True)
        selected_flow_dir = selected_flow / np.maximum(selected_flow_norm, 1e-8)
        target_xyz = target[selected, 0, :3]
        normal_xyz = normal[selected, 0, :3]
        zero_xyz = zero[selected, 0, :3]
        flip_xyz = flip[selected, 0, :3]
        per_task[task] = {
            "windows": int(np.sum(selected)),
            "normal_first_action_mse": float(np.mean((normal[selected, 0] - target[selected, 0]) ** 2)),
            "zero_first_action_mse": float(np.mean((zero[selected, 0] - target[selected, 0]) ** 2)),
            "flow_flip_first_action_mse": float(np.mean((flip[selected, 0] - target[selected, 0]) ** 2)),
            "normal_first_xyz_cosine_to_target": stats(cosine(normal_xyz, target_xyz)),
            "target_first_xyz_flow_projection": stats(np.sum(target_xyz * selected_flow_dir, axis=1)),
            "normal_first_xyz_flow_projection": stats(np.sum(normal_xyz * selected_flow_dir, axis=1)),
            "zero_first_xyz_flow_projection": stats(np.sum(zero_xyz * selected_flow_dir, axis=1)),
            "flow_flip_first_xyz_flow_projection": stats(np.sum(flip_xyz * selected_flow_dir, axis=1)),
            "normal_zero_first_xyz_delta_norm": stats(
                np.linalg.norm(normal[selected, 0, :3] - zero[selected, 0, :3], axis=1)
            ),
        }

    pair_reports: dict[str, Any] = {}
    for family, pairs in inverse_pairs.items():
        index0, index1 = pairs[:, 0], pairs[:, 1]
        target_delta = target[index0, 0, :3] - target[index1, 0, :3]
        normal_delta = normal[index0, 0, :3] - normal[index1, 0, :3]
        zero_delta = zero[index0, 0, :3] - zero[index1, 0, :3]

        base0 = base_x[index0]
        base1 = base_x[index1]
        g0 = g_x[index0]
        g1 = g_x[index1]
        same_state0_delta = (
            predictor.predict(base0, g0)[:, 0, :3]
            - predictor.predict(base0, g1)[:, 0, :3]
        )
        same_state1_delta = (
            predictor.predict(base1, g0)[:, 0, :3]
            - predictor.predict(base1, g1)[:, 0, :3]
        )
        same_state_delta = np.concatenate([same_state0_delta, same_state1_delta], axis=0)
        repeated_target_delta = np.concatenate([target_delta, target_delta], axis=0)
        pair_reports[family] = {
            "pairs": int(len(pairs)),
            "expert_delta_norm": stats(np.linalg.norm(target_delta, axis=1)),
            "normal_pair_delta_cosine_to_expert": stats(cosine(normal_delta, target_delta)),
            "zero_g_pair_delta_cosine_to_expert": stats(cosine(zero_delta, target_delta)),
            "same_state_g_only_delta_norm": stats(np.linalg.norm(same_state_delta, axis=1)),
            "same_state_g_only_delta_cosine_to_expert": stats(
                cosine(same_state_delta, repeated_target_delta)
            ),
        }

    return {
        "checkpoint": predictor.checkpoint_path.resolve().as_posix(),
        "mode": predictor.mode,
        "overall": {
            "windows": int(len(target)),
            "normal_first_action_mse": float(np.mean((normal[:, 0] - target[:, 0]) ** 2)),
            "zero_first_action_mse": float(np.mean((zero[:, 0] - target[:, 0]) ** 2)),
            "flow_flip_first_action_mse": float(np.mean((flip[:, 0] - target[:, 0]) ** 2)),
            "normal_zero_first_xyz_delta_norm": stats(
                np.linalg.norm(normal[:, 0, :3] - zero[:, 0, :3], axis=1)
            ),
        },
        "per_task": per_task,
        "inverse_pairs": pair_reports,
    }


def parse_checkpoints(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=PATH, got {value}")
        label, path = value.split("=", 1)
        output[label] = Path(path).resolve()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calvin-code", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-meta", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument(
        "--include-tasks",
        nargs="+",
        default=None,
        help="Override run_meta include_tasks for an evaluation-only task audit.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    with open(args.run_meta, encoding="utf-8") as handle:
        run_meta = json.load(handle)
    checkpoints = parse_checkpoints(args.checkpoints)
    first_checkpoint = torch.load(next(iter(checkpoints.values())), map_location="cpu")
    text_to_task, _ = base.load_eval_annotation_maps(args.calvin_code.resolve())
    task_to_id = {str(task): int(index) for task, index in run_meta["tasks"].items()}
    arrays, _ = policy.collect_chunk_records(
        args.data_root.resolve(),
        task_to_id,
        chunk_size=int(run_meta["chunk_size"]),
        max_windows=int(run_meta.get("max_windows", 0)),
        max_segments=int(run_meta.get("max_segments", 0)),
        stride=int(run_meta["stride"]),
        seed=int(run_meta["data_seed"]),
        include_tasks=(
            set(args.include_tasks)
            if args.include_tasks
            else (set(run_meta["include_tasks"]) if run_meta.get("include_tasks") else None)
        ),
        articulation_geometry=first_checkpoint["articulation_geometry"],
        minimal_g_scope=str(first_checkpoint.get("minimal_g_scope", "articulation")),
        primitive_geometry=first_checkpoint.get("primitive_geometry") or {},
        task_segment_limits=run_meta.get("task_segment_limits"),
        reverse_open_augmentation=bool(run_meta.get("reverse_open_augmentation", False)),
        reverse_open_max_segments=int(run_meta.get("reverse_open_max_segments", 0)),
        reverse_task_pairs=policy.parse_reverse_task_pairs(run_meta.get("reverse_task_pairs")),
        reverse_pair_max_segments=int(run_meta.get("reverse_pair_max_segments", 0)),
    )
    _, val_idx = policy.segment_train_val_indices(arrays["segment_id"], int(run_meta["split_seed"]))
    base_x, g_x = policy.make_minimal_query_condition_parts(arrays)
    target = arrays["action_chunks"][val_idx].astype(np.float32)
    val_base = base_x[val_idx].astype(np.float32)
    val_g = g_x[val_idx].astype(np.float32)
    val_tasks = arrays["task_name"][val_idx].astype(str)
    inverse_pairs = policy.build_inverse_goal_train_pairs(
        arrays,
        val_idx,
        max_eef_distance_m=0.05,
        max_mechanism_fraction=0.10,
    )
    report = {
        "analyzed_tasks": args.include_tasks or run_meta.get("include_tasks"),
        "validation_windows": int(len(val_idx)),
        "inverse_pair_counts": {name: int(len(pairs)) for name, pairs in inverse_pairs.items()},
        "models": {
            label: model_report(
                BatchPredictor(path, args.device, args.batch_size),
                val_base,
                val_g,
                target,
                val_tasks,
                inverse_pairs,
            )
            for label, path in checkpoints.items()
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "minimal_g_token_condition_sensitivity.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=policy.json_default)
    print(json.dumps(report, indent=2, default=policy.json_default), flush=True)


if __name__ == "__main__":
    main()
