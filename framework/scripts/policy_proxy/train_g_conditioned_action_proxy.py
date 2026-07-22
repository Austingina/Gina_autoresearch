#!/usr/bin/env python3
"""Small offline policy-proxy test for reusable G.

This does not replace a CALVIN policy rollout. It asks a narrower question:
does predicted reusable-G add action-predictive signal on a held-out task split
after controlling for no-G, zero-G, source-mean-G, shuffled-G, and oracle-G?
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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


def standardize(x: np.ndarray, eps: float = 1e-6):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (x - mean) / std, mean.astype(np.float32), std.astype(np.float32)


def load_actions(dataset_root: Path, frame_ids: np.ndarray, split: str, action_key: str) -> np.ndarray:
    out = []
    split_dir = dataset_root / split
    for fid in frame_ids.astype(int):
        path = split_dir / f"episode_{fid:07d}.npz"
        with np.load(path, allow_pickle=True) as frame:
            out.append(frame[action_key].astype(np.float32))
    return np.asarray(out, dtype=np.float32)


def subset_indices(n: int, max_records: int, seed: int) -> np.ndarray:
    if max_records <= 0 or max_records >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_records, replace=False))


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


def action_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    out = {
        "mse": float(np.mean(err**2)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "pos_mse": float(np.mean(err[:, :3] ** 2)),
        "rot_mse": float(np.mean(err[:, 3:6] ** 2)),
        "gripper_mse": float(np.mean(err[:, 6:] ** 2)),
        "gripper_sign_acc": float(np.mean(np.sign(y_pred[:, 6]) == np.sign(y_true[:, 6]))),
    }
    return out


def train_variant(
    name: str,
    train_x_raw: np.ndarray,
    val_x_raw: np.ndarray,
    train_y_raw: np.ndarray,
    val_y_raw: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    _, x_mean, x_scale = standardize(train_x_raw)
    _, y_mean, y_scale = standardize(train_y_raw)
    train_x = (train_x_raw - x_mean) / x_scale
    val_x = (val_x_raw - x_mean) / x_scale
    train_y = (train_y_raw - y_mean) / y_scale
    val_y = (val_y_raw - y_mean) / y_scale

    train_ds = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float())
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)
    model = MLP(train_x.shape[1], train_y.shape[1], args.hidden, args.depth, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    val_x_t = torch.from_numpy(val_x).float().to(device)
    val_y_t = torch.from_numpy(val_y).float().to(device)
    train_losses = []
    val_losses = []

    for _ in range(args.epochs):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
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
            val_losses.append(float(loss_fn(model(val_x_t), val_y_t).item()))

    model.eval()
    with torch.no_grad():
        val_pred_std = model(val_x_t).cpu().numpy()
    val_pred = val_pred_std * y_scale + y_mean
    metrics = {
        "variant": name,
        "feature_dim": int(train_x_raw.shape[1]),
        "action_dim": int(train_y_raw.shape[1]),
        "epochs": args.epochs,
        "final_train_mse_standardized": train_losses[-1],
        "final_val_mse_standardized": val_losses[-1],
        "best_val_mse_standardized": min(val_losses),
        **{f"val_{k}": v for k, v in action_metrics(val_y_raw, val_pred).items()},
    }

    variant_dir = out_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    write_json(variant_dir / "metrics.json", metrics)
    np.savez_compressed(
        variant_dir / "predictions.npz",
        val_y_true=val_y_raw.astype(np.float32),
        val_y_pred=val_pred.astype(np.float32),
    )
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("standardized mse")
    plt.legend()
    plt.tight_layout()
    plt.savefig(variant_dir / "loss_curve.png", dpi=160)
    plt.close()
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--source-g-dir", type=Path, required=True)
    ap.add_argument("--target-g-dir", type=Path, required=True)
    ap.add_argument("--g-predictions", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--feature-mode", default="canonical_nearest", choices=["raw", "canonical_nearest"])
    ap.add_argument("--action-key", default="rel_actions", choices=["actions", "rel_actions"])
    ap.add_argument("--split", default="training")
    ap.add_argument("--max-train-records", type=int, default=8000)
    ap.add_argument("--max-val-records", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--torch-threads", type=int, default=8)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "cpu":
        device = torch.device("cpu")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_data = np.load(args.source_g_dir / "candidate_g_labels.npz", allow_pickle=True)
    target_data = np.load(args.target_g_dir / "candidate_g_labels.npz", allow_pickle=True)
    preds = np.load(args.g_predictions, allow_pickle=True)

    train_idx = subset_indices(len(source_data["frame_ids"]), args.max_train_records, args.seed)
    val_idx = subset_indices(len(target_data["frame_ids"]), args.max_val_records, args.seed + 1)

    train_base = flatten_features(source_data, args.feature_mode)[train_idx]
    val_base = flatten_features(target_data, args.feature_mode)[val_idx]
    train_actions = load_actions(args.dataset_root, source_data["frame_ids"][train_idx], args.split, args.action_key)
    val_actions = load_actions(args.dataset_root, target_data["frame_ids"][val_idx], args.split, args.action_key)

    train_true_g = preds["train_y_true"][train_idx].astype(np.float32)
    val_true_g = preds["val_y_true"][val_idx].astype(np.float32)
    train_pred_g = preds["train_y_pred"][train_idx].astype(np.float32)
    val_pred_g = preds["val_y_pred"][val_idx].astype(np.float32)
    mean_g = train_true_g.mean(axis=0, keepdims=True).astype(np.float32)

    rng = np.random.default_rng(args.seed + 2)
    train_shuffle = train_pred_g[rng.permutation(len(train_pred_g))]
    val_shuffle = val_pred_g[rng.permutation(len(val_pred_g))]

    variants = {
        "no_g": (train_base, val_base),
        "zero_g": (
            np.concatenate([train_base, np.zeros_like(train_true_g)], axis=1),
            np.concatenate([val_base, np.zeros_like(val_true_g)], axis=1),
        ),
        "source_mean_g": (
            np.concatenate([train_base, np.broadcast_to(mean_g, train_true_g.shape)], axis=1),
            np.concatenate([val_base, np.broadcast_to(mean_g, val_true_g.shape)], axis=1),
        ),
        "shuffled_pred_g": (
            np.concatenate([train_base, train_shuffle], axis=1),
            np.concatenate([val_base, val_shuffle], axis=1),
        ),
        "predicted_g": (
            np.concatenate([train_base, train_pred_g], axis=1),
            np.concatenate([val_base, val_pred_g], axis=1),
        ),
        "oracle_g": (
            np.concatenate([train_base, train_true_g], axis=1),
            np.concatenate([val_base, val_true_g], axis=1),
        ),
    }

    run_meta = {
        "dataset_root": str(args.dataset_root),
        "source_g_dir": str(args.source_g_dir),
        "target_g_dir": str(args.target_g_dir),
        "g_predictions": str(args.g_predictions),
        "feature_mode": args.feature_mode,
        "action_key": args.action_key,
        "split": args.split,
        "train_records": int(len(train_idx)),
        "val_records": int(len(val_idx)),
        "base_feature_dim": int(train_base.shape[1]),
        "g_dim": int(train_true_g.shape[1]),
        "action_dim": int(train_actions.shape[1]),
        "device": str(device),
        "seed": args.seed,
    }
    write_json(args.out_dir / "run_meta.json", run_meta)

    rows = []
    for name, (train_x, val_x) in variants.items():
        print(f"[train] {name} train_x={train_x.shape} val_x={val_x.shape}", flush=True)
        rows.append(train_variant(name, train_x, val_x, train_actions, val_actions, args, device, args.out_dir))

    rows.sort(key=lambda r: r["val_mse"])
    with (args.out_dir / "metrics_summary.csv").open("w", newline="") as f:
        fieldnames = sorted(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.out_dir / "metrics_summary.json", {"meta": run_meta, "results": rows})
    print(json.dumps({"meta": run_meta, "results": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
