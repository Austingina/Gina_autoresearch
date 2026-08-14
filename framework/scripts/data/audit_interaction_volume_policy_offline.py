#!/usr/bin/env python3
"""Audit matched no-G and interaction-volume-G policies on CALVIN demonstrations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
POLICY_DIR = SCRIPT_DIR.parent / "policy_state_bc"
sys.path.insert(0, SCRIPT_DIR.as_posix())
sys.path.insert(0, POLICY_DIR.as_posix())

import train_eval_state_chunk_flow_policy as policy  # noqa: E402
from build_calvin_interaction_volume_cache import (  # noqa: E402
    load_clearance,
    relabel_segment,
)
from calibrate_calvin_block_surface_g import (  # noqa: E402
    DEFAULT_HELDOUT_TASKS,
    DEFAULT_SOURCE_TASKS,
    TASKS,
    geometry_for_segment,
    load_scene_geometries,
    load_segments,
)


CONTROLS = (
    "normal",
    "zero",
    "sign_flip",
    "zero_attraction",
    "zero_flow",
)


def load_model(checkpoint: Path, device: torch.device) -> tuple[dict[str, object], torch.nn.Module]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    mode = str(ckpt["mode"])
    if mode == "chunk_bc_state":
        model: torch.nn.Module = policy.ChunkMLP(
            int(ckpt["cond_dim"]),
            int(ckpt["action_dim"]),
            int(ckpt["hidden"]),
            int(ckpt["layers"]),
        )
    elif mode in {
        "chunk_bc_minimal_g_tokens",
        "chunk_bc_minimal_g_factorized_tokens",
    }:
        model_class = (
            policy.FactorizedMinimalGTokenChunkPolicy
            if mode == "chunk_bc_minimal_g_factorized_tokens"
            else policy.MinimalGTokenChunkTransformer
        )
        model = model_class(
            int(ckpt["base_dim"]),
            int(ckpt["g_dim"]),
            int(ckpt["chunk_size"]),
            int(ckpt["hidden"]),
            int(ckpt["layers"]),
        )
    elif mode == "chunk_bc_minimal_g_frame":
        model = policy.FrameMinimalGChunkMLP(
            int(ckpt["base_dim"]),
            int(ckpt["g_dim"]),
            int(ckpt["action_dim"]),
            int(ckpt["chunk_size"]),
            int(ckpt["hidden"]),
            int(ckpt["layers"]),
            np.asarray(ckpt["y_mean"], dtype=np.float32),
            np.asarray(ckpt["y_std"], dtype=np.float32),
        )
    else:
        raise RuntimeError(f"Unsupported checkpoint mode for this audit: {mode}")
    model.load_state_dict(ckpt["model_state"])
    return ckpt, model.to(device).eval()


def predict_first_actions(
    ckpt: dict[str, object],
    model: torch.nn.Module,
    robot_obs: np.ndarray,
    minimal_g: np.ndarray,
    control: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    mode = str(ckpt["mode"])
    g = policy.apply_minimal_g_control(minimal_g, control)
    cond = (
        robot_obs.astype(np.float32)
        if mode == "chunk_bc_state"
        else np.concatenate([robot_obs, g], axis=1).astype(np.float32)
    )
    cond_n = (cond - np.asarray(ckpt["cond_mean"])) / np.asarray(ckpt["cond_std"])
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(cond_n), batch_size):
            stop = min(len(cond_n), start + batch_size)
            cond_t = torch.from_numpy(cond_n[start:stop].astype(np.float32)).to(device)
            if mode == "chunk_bc_state":
                pred_n = model(cond_t)
            else:
                g_t = torch.from_numpy(g[start:stop].astype(np.float32)).to(device)
                pred_n = model(cond_t[:, : int(ckpt["base_dim"])], g_t)
            raw = (
                pred_n.cpu().numpy() * np.asarray(ckpt["y_std"])
                + np.asarray(ckpt["y_mean"])
            ).reshape(stop - start, int(ckpt["chunk_size"]), 7)
            outputs.append(raw[:, 0])
    return np.concatenate(outputs).astype(np.float32)


def metric_summary(
    prediction: np.ndarray,
    target: np.ndarray,
    flow: np.ndarray,
    attraction: np.ndarray,
) -> dict[str, float | int]:
    error = prediction - target
    xyz_error = error[:, :3]
    flow_parallel_error = np.sum(xyz_error * flow, axis=1, keepdims=True) * flow
    flow_orthogonal_error = xyz_error - flow_parallel_error
    attraction_norm = np.linalg.norm(attraction, axis=1, keepdims=True)
    attraction_direction = attraction / np.maximum(attraction_norm, 1e-6)
    attraction_valid = attraction_norm[:, 0] > 1e-6
    predicted_attraction_projection = (
        float(
            np.mean(
                np.sum(
                    prediction[attraction_valid, :3]
                    * attraction_direction[attraction_valid],
                    axis=1,
                )
            )
        )
        if np.any(attraction_valid)
        else None
    )
    target_attraction_projection = (
        float(
            np.mean(
                np.sum(
                    target[attraction_valid, :3]
                    * attraction_direction[attraction_valid],
                    axis=1,
                )
            )
        )
        if np.any(attraction_valid)
        else None
    )
    return {
        "windows": int(len(prediction)),
        "first_action_mse": float(np.mean(error**2)),
        "first_xyz_mse": float(np.mean(error[:, :3] ** 2)),
        "first_x_mse": float(np.mean(error[:, 0] ** 2)),
        "first_y_mse": float(np.mean(error[:, 1] ** 2)),
        "first_z_mse": float(np.mean(error[:, 2] ** 2)),
        "first_xyz_flow_parallel_mse": float(np.mean(flow_parallel_error**2)),
        "first_xyz_flow_orthogonal_mse": float(np.mean(flow_orthogonal_error**2)),
        "first_rotation_mse": float(np.mean(error[:, 3:6] ** 2)),
        "first_gripper_mse": float(np.mean(error[:, 6] ** 2)),
        "first_gripper_accuracy": float(
            np.mean(np.sign(prediction[:, -1]) == np.sign(target[:, -1]))
        ),
        "predicted_flow_projection": float(
            np.mean(np.sum(prediction[:, :3] * flow, axis=1))
        ),
        "target_flow_projection": float(
            np.mean(np.sum(target[:, :3] * flow, axis=1))
        ),
        "nonzero_attraction_windows": int(np.sum(attraction_valid)),
        "predicted_attraction_projection": predicted_attraction_projection,
        "target_attraction_projection": target_attraction_projection,
        "first_xyz_attraction_parallel_mse": (
            float(
                np.mean(
                    np.sum(
                        xyz_error[attraction_valid] * attraction_direction[attraction_valid],
                        axis=1,
                        keepdims=True,
                    )
                    ** 2
                )
                / 3.0
            )
            if np.any(attraction_valid)
            else None
        ),
        "first_xyz_attraction_orthogonal_mse": (
            float(
                np.mean(
                    (
                        xyz_error[attraction_valid]
                        - np.sum(
                            xyz_error[attraction_valid]
                            * attraction_direction[attraction_valid],
                            axis=1,
                            keepdims=True,
                        )
                        * attraction_direction[attraction_valid]
                    )
                    ** 2
                )
            )
            if np.any(attraction_valid)
            else None
        ),
        "predicted_xyz_norm": float(np.mean(np.linalg.norm(prediction[:, :3], axis=1))),
        "target_xyz_norm": float(np.mean(np.linalg.norm(target[:, :3], axis=1))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--calvin-code", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--no-g-checkpoint", type=Path, required=True)
    parser.add_argument("--g-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-threshold", type=float, default=0.002)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    data_root = args.data_root.resolve()
    split_dir = data_root / "training"
    clearance = load_clearance(args.calibration_report.resolve())
    geometries = load_scene_geometries(
        split_dir, args.calvin_code.resolve() / "calvin_env"
    )
    no_g_ckpt, no_g_model = load_model(args.no_g_checkpoint.resolve(), device)
    g_ckpt, g_model = load_model(args.g_checkpoint.resolve(), device)
    if int(no_g_ckpt["chunk_size"]) != int(g_ckpt["chunk_size"]):
        raise RuntimeError("No-G and G checkpoints use different chunk sizes")
    chunk_size = int(g_ckpt["chunk_size"])

    records: dict[str, list[np.ndarray | str]] = defaultdict(list)
    segment_rows = []
    for task, start, end in load_segments(split_dir):
        if task not in DEFAULT_SOURCE_TASKS + DEFAULT_HELDOUT_TASKS:
            continue
        arrays, metadata = relabel_segment(
            split_dir,
            task,
            start,
            end,
            geometry_for_segment(geometries, start, end),
            clearance,
            args.motion_threshold,
        )
        onset = int(metadata["motion_onset_local_index"])
        windows = np.arange(0, len(arrays["frame_index"]) - chunk_size + 1, args.stride)
        attraction_norm = np.linalg.norm(arrays["minimal_g"][:, :3], axis=1)
        phases = np.asarray(
            [
                "post_motion"
                if index >= onset
                else "pre_inside_volume"
                if attraction_norm[index] <= 1e-7
                else "pre_outside_volume"
                for index in windows
            ],
            dtype=object,
        )
        split = "source" if task in DEFAULT_SOURCE_TASKS else "heldout"
        records["robot_obs"].append(arrays["robot_obs"][windows])
        records["minimal_g"].append(arrays["minimal_g"][windows])
        records["target"].append(arrays["rel_actions"][windows])
        records["task"].append(np.full(len(windows), task, dtype=object))
        records["split"].append(np.full(len(windows), split, dtype=object))
        records["phase"].append(phases)
        segment_rows.append(metadata)

    joined = {
        key: np.concatenate(value) for key, value in records.items()
    }
    predictions = {
        "no_g": {
            "normal": predict_first_actions(
                no_g_ckpt,
                no_g_model,
                joined["robot_obs"],
                joined["minimal_g"],
                "normal",
                device,
                args.batch_size,
            )
        },
        "volume_g": {
            control: predict_first_actions(
                g_ckpt,
                g_model,
                joined["robot_obs"],
                joined["minimal_g"],
                control,
                device,
                args.batch_size,
            )
            for control in CONTROLS
        },
    }

    groups = {
        "all": np.ones(len(joined["target"]), dtype=bool),
        **{
            f"split/{split}": joined["split"] == split
            for split in ("source", "heldout")
        },
        **{
            f"task/{task}": joined["task"] == task
            for task in DEFAULT_SOURCE_TASKS + DEFAULT_HELDOUT_TASKS
        },
        **{
            f"task_phase/{task}/{phase}": (joined["task"] == task)
            & (joined["phase"] == phase)
            for task in DEFAULT_SOURCE_TASKS + DEFAULT_HELDOUT_TASKS
            for phase in (
                "pre_outside_volume",
                "pre_inside_volume",
                "post_motion",
            )
        },
        **{
            f"split_phase/{split}/{phase}": (joined["split"] == split)
            & (joined["phase"] == phase)
            for split in ("source", "heldout")
            for phase in (
                "pre_outside_volume",
                "pre_inside_volume",
                "post_motion",
            )
        },
    }
    report_groups: dict[str, object] = {}
    for name, mask in groups.items():
        if not np.any(mask):
            continue
        flow = joined["minimal_g"][mask, 3:6]
        flow = flow / np.maximum(np.linalg.norm(flow, axis=1, keepdims=True), 1e-6)
        attraction = joined["minimal_g"][mask, :3]
        group_metrics: dict[str, object] = {}
        for model_name, model_predictions in predictions.items():
            group_metrics[model_name] = {
                control: metric_summary(
                    prediction[mask], joined["target"][mask], flow, attraction
                )
                for control, prediction in model_predictions.items()
            }
        normal = predictions["volume_g"]["normal"][mask]
        zero = predictions["volume_g"]["zero"][mask]
        flip = predictions["volume_g"]["sign_flip"][mask]
        normal_zero = normal - zero
        normal_flip = normal - flip
        group_metrics["volume_g_causal_response"] = {
            "normal_zero_action_l2": float(
                np.mean(np.linalg.norm(normal_zero, axis=1))
            ),
            "normal_flip_action_l2": float(
                np.mean(np.linalg.norm(normal_flip, axis=1))
            ),
            "normal_zero_xyz_l2": float(
                np.mean(np.linalg.norm(normal_zero[:, :3], axis=1))
            ),
            "normal_flip_xyz_l2": float(
                np.mean(np.linalg.norm(normal_flip[:, :3], axis=1))
            ),
            "normal_zero_rotation_l2": float(
                np.mean(np.linalg.norm(normal_zero[:, 3:6], axis=1))
            ),
            "normal_flip_rotation_l2": float(
                np.mean(np.linalg.norm(normal_flip[:, 3:6], axis=1))
            ),
            "normal_zero_gripper_abs": float(np.mean(np.abs(normal_zero[:, 6]))),
            "normal_flip_gripper_abs": float(np.mean(np.abs(normal_flip[:, 6]))),
            "normal_minus_zero_flow_projection": float(
                np.mean(np.sum(normal_zero[:, :3] * flow, axis=1))
            ),
            "normal_minus_flip_flow_projection": float(
                np.mean(np.sum(normal_flip[:, :3] * flow, axis=1))
            ),
            "normal_minus_zero_attraction_projection": (
                float(
                    np.mean(
                        np.sum(
                            normal_zero[:, :3]
                            * (
                                attraction
                                / np.maximum(
                                    np.linalg.norm(attraction, axis=1, keepdims=True),
                                    1e-6,
                                )
                            ),
                            axis=1,
                        )[np.linalg.norm(attraction, axis=1) > 1e-6]
                    )
                )
                if np.any(np.linalg.norm(attraction, axis=1) > 1e-6)
                else None
            ),
        }
        report_groups[name] = group_metrics

    report = {
        "data_root": data_root.as_posix(),
        "calibration_report": args.calibration_report.resolve().as_posix(),
        "no_g_checkpoint": args.no_g_checkpoint.resolve().as_posix(),
        "g_checkpoint": args.g_checkpoint.resolve().as_posix(),
        "tool_clearance": clearance,
        "motion_threshold": args.motion_threshold,
        "stride": args.stride,
        "chunk_size": chunk_size,
        "segments": len(segment_rows),
        "windows": len(joined["target"]),
        "groups": report_groups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
