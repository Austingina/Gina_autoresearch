#!/usr/bin/env python3
"""Train a tiny future-G predictor on extracted candidate CALVIN G labels."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def timestamp() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def nearest_indices(data: np.lib.npyio.NpzFile) -> np.ndarray:
    object_pos = data["object_pos"]
    nearest = data["nearest_object_index"].astype(np.int64).reshape(-1)
    return np.clip(nearest, 0, object_pos.shape[1] - 1)


def take_nearest(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    arr = data[key]
    return arr[np.arange(len(arr)), nearest_indices(data)]


def flatten_features(data: np.lib.npyio.NpzFile, feature_mode: str) -> np.ndarray:
    if feature_mode == "raw":
        parts = [
            data["eef_pos"],
            data["eef_orn"],
            data["gripper"],
            data["scene_scalars"],
            data["object_pos"].reshape(len(data["eef_pos"]), -1),
            data["object_orn"].reshape(len(data["eef_pos"]), -1),
            data["rel_object_to_eef"].reshape(len(data["eef_pos"]), -1),
            data["dist_object_to_eef"],
            data["near_contact"],
        ]
    elif feature_mode == "canonical_nearest":
        nearest = nearest_indices(data)
        parts = [
            data["eef_pos"],
            data["eef_orn"],
            data["gripper"],
            data["scene_scalars"],
            take_nearest(data, "object_pos"),
            take_nearest(data, "object_orn"),
            take_nearest(data, "rel_object_to_eef"),
            data["dist_object_to_eef"][np.arange(len(nearest)), nearest][:, None],
            data["near_contact"][np.arange(len(nearest)), nearest][:, None],
        ]
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")
    return np.concatenate([p.astype(np.float32) for p in parts], axis=1)


def future_scene_delta(data: np.lib.npyio.NpzFile, horizon: int) -> np.ndarray:
    key = f"future_delta_scene_scalars_h{horizon}"
    if key not in data:
        raise KeyError(
            f"{key} is missing. Re-run extract_candidate_g_labels.py after the G-v2 scene-delta update."
        )
    return data[key].astype(np.float32)


def nearest_object_delta(data: np.lib.npyio.NpzFile, horizon: int) -> np.ndarray:
    obj = data[f"future_delta_object_pos_h{horizon}"].astype(np.float32)
    nearest = nearest_indices(data)
    return obj[np.arange(len(obj)), nearest]


def flatten_targets(data: np.lib.npyio.NpzFile, horizon: int, target_mode: str) -> np.ndarray:
    eef = data[f"future_delta_eef_h{horizon}"].astype(np.float32)
    obj = data[f"future_delta_object_pos_h{horizon}"].reshape(len(eef), -1).astype(np.float32)
    if target_mode == "raw":
        return np.concatenate([eef, obj], axis=1)
    if target_mode == "raw_with_scene":
        scene = future_scene_delta(data, horizon)
        return np.concatenate([eef, obj, scene], axis=1)
    if target_mode == "scene_only":
        return future_scene_delta(data, horizon)
    if target_mode == "canonical_nearest":
        return np.concatenate([eef, nearest_object_delta(data, horizon)], axis=1)
    if target_mode == "canonical_nearest_with_scene":
        scene = future_scene_delta(data, horizon)
        return np.concatenate([eef, nearest_object_delta(data, horizon), scene], axis=1)
    raise ValueError(f"Unknown target_mode: {target_mode}")


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, depth: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(depth):
            layers.extend([nn.Linear(dim, hidden), nn.ReLU()])
            if dropout:
                layers.append(nn.Dropout(dropout))
            dim = hidden
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def standardize(x: np.ndarray, eps: float = 1e-6):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (x - mean) / std, mean.astype(np.float32), std.astype(np.float32)


def plot_losses(out_dir: Path, train_losses: list[float], val_losses: list[float]) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("mse")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=160)
    plt.close()


def plot_predictions(out_dir: Path, y_true: np.ndarray, y_pred: np.ndarray, max_dims: int = 12) -> None:
    dims = min(max_dims, y_true.shape[1])
    fig, axes = plt.subplots(dims, 1, figsize=(8, max(2, dims * 1.2)), sharex=True)
    if dims == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(y_true[:, i], label="gt", linewidth=1.5)
        ax.plot(y_pred[:, i], label="pred", linewidth=1.0)
        ax.set_ylabel(f"y{i}")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("val sample")
    plt.tight_layout()
    plt.savefig(out_dir / "prediction_preview.png", dpi=160)
    plt.close()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    max_abs = float(np.max(np.abs(err))) if err.size else 0.0
    return {
        "mse_raw": mse,
        "rmse_raw": rmse,
        "mae_raw": mae,
        "max_abs_raw": max_abs,
    }


def clip_targets(
    train_y: np.ndarray,
    val_y: np.ndarray,
    quantile: float,
    min_abs: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if quantile <= 0:
        return train_y, val_y, None
    if not 0.0 < quantile <= 1.0:
        raise ValueError(f"target_clip_quantile must be in (0, 1], got {quantile}")
    clip_abs = np.quantile(np.abs(train_y), quantile, axis=0, keepdims=True).astype(np.float32)
    clip_abs = np.maximum(clip_abs, min_abs)
    return (
        np.clip(train_y, -clip_abs, clip_abs).astype(np.float32),
        np.clip(val_y, -clip_abs, clip_abs).astype(np.float32),
        clip_abs,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g-dir", type=Path, required=True)
    ap.add_argument(
        "--val-g-dir",
        type=Path,
        default=None,
        help="Optional held-out G label directory. If set, train on --g-dir and validate on this directory.",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--feature-mode",
        default="raw",
        choices=["raw", "canonical_nearest"],
        help=(
            "Input feature layout. raw keeps V1 named object slots; canonical_nearest "
            "uses only the nearest interaction object features and omits object identity."
        ),
    )
    ap.add_argument(
        "--target-mode",
        default="raw",
        choices=[
            "raw",
            "raw_with_scene",
            "scene_only",
            "canonical_nearest",
            "canonical_nearest_with_scene",
        ],
        help=(
            "Prediction target layout. raw keeps V1 eef+all object deltas; "
            "raw_with_scene adds scene scalar deltas; canonical_nearest predicts only "
            "the nearest object delta; canonical_nearest_with_scene combines nearest "
            "object and scene scalar deltas."
        ),
    )
    ap.add_argument(
        "--target-clip-quantile",
        type=float,
        default=0.0,
        help="If >0, clip target dimensions by the source/train absolute quantile before training and metrics.",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = args.out_dir or (
        Path("/mnt/nvme33n1/wuk_code/AutoResearch")
        / "artifacts"
        / "g_predictors"
        / f"{timestamp()}_candidate_g_h{args.horizon}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data = np.load(args.g_dir / "candidate_g_labels.npz", allow_pickle=True)
    train_x_raw = flatten_features(train_data, args.feature_mode)
    train_y_raw = flatten_targets(train_data, args.horizon, args.target_mode)
    if args.val_g_dir is not None:
        val_data = np.load(args.val_g_dir / "candidate_g_labels.npz", allow_pickle=True)
        val_x_raw = flatten_features(val_data, args.feature_mode)
        val_y_raw = flatten_targets(val_data, args.horizon, args.target_mode)
        train_idx = np.arange(len(train_x_raw))
        val_idx = np.arange(len(val_x_raw))
        train_frame_ids = train_data["frame_ids"]
        train_sequence_index = train_data["sequence_index"]
        val_frame_ids = val_data["frame_ids"]
        val_sequence_index = val_data["sequence_index"]
        split_mode = "heldout_g_dir"
    else:
        val_data = train_data
        x = train_x_raw
        y = train_y_raw
        n = len(x)
        if n < 8:
            raise ValueError(f"Need at least 8 records, got {n}")
        order = np.random.permutation(n)
        val_n = max(1, int(round(n * args.val_ratio)))
        val_idx = order[:val_n]
        train_idx = order[val_n:]
        train_x_raw = x[train_idx]
        train_y_raw = y[train_idx]
        val_x_raw = x[val_idx]
        val_y_raw = y[val_idx]
        train_frame_ids = train_data["frame_ids"][train_idx]
        train_sequence_index = train_data["sequence_index"][train_idx]
        val_frame_ids = train_data["frame_ids"][val_idx]
        val_sequence_index = train_data["sequence_index"][val_idx]
        split_mode = "random_record_split"

    if len(train_x_raw) < 4:
        raise ValueError(f"Need at least 4 train records, got {len(train_x_raw)}")
    if len(val_x_raw) < 1:
        raise ValueError("Need at least 1 validation record")

    train_y_raw, val_y_raw, target_clip_abs = clip_targets(
        train_y_raw,
        val_y_raw,
        args.target_clip_quantile,
    )

    _, x_mean, x_scale = standardize(train_x_raw)
    _, y_mean, y_scale = standardize(train_y_raw)
    train_x = (train_x_raw - x_mean) / x_scale
    train_y = (train_y_raw - y_mean) / y_scale
    val_x_np = (val_x_raw - x_mean) / x_scale
    val_y_np = (val_y_raw - y_mean) / y_scale

    train_ds = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float())
    val_x = torch.from_numpy(val_x_np).float()
    val_y = torch.from_numpy(val_y_np).float()
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = MLP(train_x_raw.shape[1], train_y_raw.shape[1], args.hidden, args.depth, args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    train_losses: list[float] = []
    val_losses: list[float] = []

    for _ in range(args.epochs):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            count += len(xb)
        train_losses.append(total / max(count, 1))
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y).item())
        val_losses.append(val_loss)

    model.eval()
    with torch.no_grad():
        train_pred_std = model(torch.from_numpy(train_x).float()).numpy()
        val_pred_std = model(torch.from_numpy(val_x_np).float()).numpy()
    train_pred = train_pred_std * y_scale + y_mean
    val_pred = val_pred_std * y_scale + y_mean
    np.savez_compressed(
        out_dir / "predictions.npz",
        train_y_true=train_y_raw.astype(np.float32),
        train_y_pred=train_pred.astype(np.float32),
        val_y_true=val_y_raw.astype(np.float32),
        val_y_pred=val_pred.astype(np.float32),
        train_idx=train_idx,
        val_idx=val_idx,
        train_frame_ids=train_frame_ids,
        train_sequence_index=train_sequence_index,
        val_frame_ids=val_frame_ids,
        val_sequence_index=val_sequence_index,
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "x_mean": x_mean,
            "x_scale": x_scale,
            "y_mean": y_mean,
            "y_scale": y_scale,
            "args": vars(args),
        },
        out_dir / "checkpoint.pt",
    )
    plot_losses(out_dir, train_losses, val_losses)
    plot_predictions(out_dir, val_y_raw, val_pred)
    train_raw_metrics = {f"train_{k}": v for k, v in regression_metrics(train_y_raw, train_pred).items()}
    val_raw_metrics = {f"val_{k}": v for k, v in regression_metrics(val_y_raw, val_pred).items()}
    train_mean = np.broadcast_to(train_y_raw.mean(axis=0, keepdims=True), train_y_raw.shape)
    val_mean = np.broadcast_to(train_y_raw.mean(axis=0, keepdims=True), val_y_raw.shape)
    train_zero_metrics = {
        f"zero_train_{k}": v
        for k, v in regression_metrics(train_y_raw, np.zeros_like(train_y_raw)).items()
    }
    val_zero_metrics = {
        f"zero_val_{k}": v
        for k, v in regression_metrics(val_y_raw, np.zeros_like(val_y_raw)).items()
    }
    train_mean_metrics = {
        f"source_mean_train_{k}": v
        for k, v in regression_metrics(train_y_raw, train_mean).items()
    }
    val_mean_metrics = {
        f"source_mean_val_{k}": v
        for k, v in regression_metrics(val_y_raw, val_mean).items()
    }
    metrics = {
        "script": "train_candidate_g_predictor.py",
        "g_dir": str(args.g_dir),
        "val_g_dir": str(args.val_g_dir) if args.val_g_dir is not None else None,
        "split_mode": split_mode,
        "horizon": args.horizon,
        "feature_mode": args.feature_mode,
        "target_mode": args.target_mode,
        "target_clip_quantile": args.target_clip_quantile,
        "target_clip_abs_mean": float(target_clip_abs.mean()) if target_clip_abs is not None else None,
        "target_clip_abs_max": float(target_clip_abs.max()) if target_clip_abs is not None else None,
        "train_records": int(len(train_x_raw)),
        "val_records": int(len(val_x_raw)),
        "feature_dim": int(train_x_raw.shape[1]),
        "target_dim": int(train_y_raw.shape[1]),
        "final_train_mse_standardized": train_losses[-1],
        "final_val_mse_standardized": val_losses[-1],
        "best_val_mse_standardized": min(val_losses),
        **train_raw_metrics,
        **val_raw_metrics,
        **train_zero_metrics,
        **val_zero_metrics,
        **train_mean_metrics,
        **val_mean_metrics,
    }
    write_json(out_dir / "metrics.json", metrics)
    (out_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Candidate G Predictor Smoke",
                "",
                f"- g_dir: `{args.g_dir}`",
                f"- val_g_dir: `{args.val_g_dir}`",
                f"- split_mode: {split_mode}",
                f"- horizon: {args.horizon}",
                f"- feature_mode: {args.feature_mode}",
                f"- target_mode: {args.target_mode}",
                f"- target_clip_quantile: {args.target_clip_quantile}",
                f"- train/val: {len(train_x_raw)}/{len(val_x_raw)}",
                f"- feature_dim: {train_x_raw.shape[1]}",
                f"- target_dim: {train_y_raw.shape[1]}",
                f"- final_train_mse_standardized: {train_losses[-1]:.6f}",
                f"- final_val_mse_standardized: {val_losses[-1]:.6f}",
                f"- best_val_mse_standardized: {min(val_losses):.6f}",
                f"- train_rmse_raw: {metrics['train_rmse_raw']:.6f}",
                f"- val_rmse_raw: {metrics['val_rmse_raw']:.6f}",
                f"- train_mae_raw: {metrics['train_mae_raw']:.6f}",
                f"- val_mae_raw: {metrics['val_mae_raw']:.6f}",
                f"- zero_train_rmse_raw: {metrics['zero_train_rmse_raw']:.6f}",
                f"- zero_val_rmse_raw: {metrics['zero_val_rmse_raw']:.6f}",
                f"- source_mean_train_rmse_raw: {metrics['source_mean_train_rmse_raw']:.6f}",
                f"- source_mean_val_rmse_raw: {metrics['source_mean_val_rmse_raw']:.6f}",
            ]
        ),
        encoding="utf-8",
    )
    print(out_dir)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
