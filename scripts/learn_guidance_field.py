#!/usr/bin/env python3
"""
Step 1 — Learn guidance field G_θ from demonstration mask trajectories.

Supervision: G*_t = normalize(centroid(M_{t+1}) - centroid(M_t)) on post-grasp steps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment import ExperimentConfig  # noqa: E402
from goals.field_data import (  # noqa: E402
    build_field_indices_from_demo_dir,
    load_demo_dir,
    train_val_split,
    train_val_split_indices,
)
from goals.field_model import MaskFieldCNN, cosine_alignment, load_field_checkpoint, save_field_checkpoint  # noqa: E402
from goals.paths import (  # noqa: E402
    assert_writable_data_path,
    default_demo_dir,
    ensure_workspace_dirs,
    field_checkpoint,
)


def _psi_from_phase(phase: float) -> np.ndarray:
    """ψ conditioning: contact_latched (1.0 if grasped/ACTUATE else 0.0)."""
    return np.array([1.0 if float(phase) >= 0.5 else 0.0], dtype=np.float32)


class MaskFieldDataset(Dataset):
    def __init__(self, samples, *, g_phase: bool = False):
        self.samples = samples
        self.g_phase = g_phase

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return torch.from_numpy(s.visual), torch.from_numpy(s.target), torch.from_numpy(_psi_from_phase(s.phase))


class LazyMaskFieldDataset(Dataset):
    def __init__(self, indices, *, image_size: int, input_mode: str, g_phase: bool = False):
        self.indices = indices
        self.image_size = image_size
        self.input_mode = input_mode
        self.g_phase = g_phase
        self._files: dict = {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        from goals.field_data import read_field_visual_at_index

        entry = self.indices[idx]
        visual = read_field_visual_at_index(
            entry, image_size=self.image_size, input_mode=self.input_mode, files=self._files
        )
        return torch.from_numpy(visual), torch.from_numpy(entry.target), torch.from_numpy(_psi_from_phase(entry.phase))


def main() -> int:
    p = argparse.ArgumentParser(description="Step 1: learn guidance field from demos")
    p.add_argument("--task", default="lift")
    p.add_argument(
        "--stage",
        default="s0_fixed",
        choices=["s0_fixed", "s1_random_cube", "s2_random_goal", "s3_ood_object"],
    )
    p.add_argument("--demo-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--supervision", default=None, choices=["state_goal", "mask_flow", "subgoal"],
                   help="S0 default state_goal; S1 default subgoal (phase-aware EE→cube / cube→goal)")
    p.add_argument(
        "--input-mode",
        default="mask_depth",
        choices=["mask", "mask_depth"],
        help="G_θ input: mask-only (6ch) or mask+depth (8ch, matches student vision).",
    )
    p.add_argument("--allow-external-data", action="store_true")
    p.add_argument(
        "--eager-load",
        action="store_true",
        help="Preload all mask frames into RAM (fast on small sets; OOM on 240px S1).",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--out-dim",
        type=int,
        default=None,
        choices=[2, 3],
        help="G_θ output dim: 2 (legacy x-z) or 3 (Interaction Geometry ℝ³). S1 subgoal default 3.",
    )
    p.add_argument(
        "--g-phase",
        action="store_true",
        help="Condition G_θ on ψ (contact_latched) — G-phase ablation vs G-impl.",
    )
    p.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional field checkpoint used to initialize model weights before fine-tuning.",
    )
    args = p.parse_args()

    if args.supervision is None:
        args.supervision = "subgoal" if args.stage == "s1_random_cube" else "state_goal"
    if args.out_dim is None:
        args.out_dim = 3 if args.supervision == "subgoal" else 2
    # subgoal needs pre-grasp (REACH) + post-grasp (ACTUATE); only s0 state_goal is post-grasp-only.
    post_grasp_only = args.supervision == "state_goal" and args.stage == "s0_fixed"
    in_channels = 8 if args.input_mode == "mask_depth" else 6
    cond_dim = 1 if args.g_phase else 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ensure_workspace_dirs(args.task, args.stage)
    demo_dir = args.demo_dir or default_demo_dir(args.task, args.stage)
    output = args.output or field_checkpoint(args.task, args.stage)

    try:
        assert_writable_data_path(demo_dir, allow_external=args.allow_external_data)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    if not demo_dir.is_dir():
        print(f"[ERROR] demo-dir not found: {demo_dir}", file=sys.stderr)
        return 1

    print(
        f"[INFO] Loading demos from {demo_dir} "
        f"(supervision={args.supervision}, input_mode={args.input_mode}, post_grasp_only={post_grasp_only}, "
        f"out_dim={args.out_dim}, g_phase={args.g_phase}, eager_load={args.eager_load})"
    )
    if args.eager_load:
        samples = load_demo_dir(
            demo_dir,
            image_size=args.image_size,
            supervision=args.supervision,
            post_grasp_only=post_grasp_only,
            input_mode=args.input_mode,
            out_dim=args.out_dim,
        )
        if len(samples) < 32:
            print(f"[ERROR] Too few field samples ({len(samples)}). Import more demos.", file=sys.stderr)
            return 1
        train_samples, val_samples = train_val_split(samples, val_ratio=args.val_ratio, seed=args.seed)
        print(f"[INFO] samples train={len(train_samples)} val={len(val_samples)}")
        train_loader = DataLoader(
            MaskFieldDataset(train_samples, g_phase=args.g_phase), batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            MaskFieldDataset(val_samples, g_phase=args.g_phase), batch_size=args.batch_size, shuffle=False
        )
        num_train, num_val = len(train_samples), len(val_samples)
    else:
        indices = build_field_indices_from_demo_dir(
            demo_dir,
            supervision=args.supervision,
            post_grasp_only=post_grasp_only,
            input_mode=args.input_mode,
            out_dim=args.out_dim,
        )
        if len(indices) < 32:
            print(f"[ERROR] Too few field samples ({len(indices)}). Import more demos.", file=sys.stderr)
            return 1
        train_idx, val_idx = train_val_split_indices(indices, val_ratio=args.val_ratio, seed=args.seed)
        print(f"[INFO] samples train={len(train_idx)} val={len(val_idx)} (lazy HDF5)")
        train_loader = DataLoader(
            LazyMaskFieldDataset(train_idx, image_size=args.image_size, input_mode=args.input_mode, g_phase=args.g_phase),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )
        val_loader = DataLoader(
            LazyMaskFieldDataset(val_idx, image_size=args.image_size, input_mode=args.input_mode, g_phase=args.g_phase),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )
        num_train, num_val = len(train_idx), len(val_idx)

    model = MaskFieldCNN(in_channels=in_channels, out_dim=args.out_dim, cond_dim=cond_dim).to(args.device)
    init_meta = None
    if args.init_checkpoint is not None:
        init_model, init_meta = load_field_checkpoint(args.init_checkpoint, device=args.device)
        if (
            init_model.in_channels != in_channels
            or init_model.out_dim != args.out_dim
            or init_model.cond_dim != cond_dim
        ):
            raise ValueError(
                "Init checkpoint architecture mismatch: "
                f"ckpt=(in={init_model.in_channels}, out={init_model.out_dim}, cond={init_model.cond_dim}) "
                f"requested=(in={in_channels}, out={args.out_dim}, cond={cond_dim})"
            )
        model.load_state_dict(init_model.state_dict())
        print(f"[INFO] Initialized field model from {args.init_checkpoint}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def _cond(cond_batch):
        return cond_batch.to(args.device) if cond_dim > 0 else None

    best_align = -1.0
    best_epoch = -1
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_align = 0.0
        n_batches = 0
        for masks, targets, cond in train_loader:
            masks = masks.to(args.device)
            targets = targets.to(args.device)
            pred = model(masks, _cond(cond))
            loss = 1.0 - cosine_alignment(pred, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += float(loss.item())
            train_align += float(cosine_alignment(pred, targets).item())
            n_batches += 1

        model.eval()
        val_align = 0.0
        val_n = 0
        # Phase-split alignment: ψ=0 REACH (pre-grasp), ψ=1 ACTUATE (post-grasp).
        reach_cos_sum, reach_n = 0.0, 0
        act_cos_sum, act_n = 0.0, 0
        with torch.no_grad():
            for masks, targets, cond in val_loader:
                masks = masks.to(args.device)
                targets = targets.to(args.device)
                pred = model(masks, _cond(cond))
                psi = cond.to(args.device).reshape(-1)
                cos = F.cosine_similarity(pred, targets, dim=-1)
                reach_mask = psi < 0.5
                if reach_mask.any():
                    reach_cos_sum += float(cos[reach_mask].sum().item())
                    reach_n += int(reach_mask.sum().item())
                if (~reach_mask).any():
                    act_cos_sum += float(cos[~reach_mask].sum().item())
                    act_n += int((~reach_mask).sum().item())
                val_align += float(cosine_alignment(pred, targets).item())
                val_n += 1
        val_align /= max(val_n, 1)
        train_loss /= max(n_batches, 1)
        train_align /= max(n_batches, 1)

        reach_align = reach_cos_sum / max(reach_n, 1)
        act_align = act_cos_sum / max(act_n, 1)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_align": train_align,
            "val_align": val_align,
            "val_reach_align": reach_align,
            "val_actuate_align": act_align,
        }
        history.append(row)
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"epoch {epoch:03d}  loss={train_loss:.4f}  train_align={train_align:.4f}  "
                f"val_align={val_align:.4f}  REACH={reach_align:.4f}({reach_n})  ACTUATE={act_align:.4f}({act_n})"
            )

        if val_align > best_align:
            best_align = val_align
            best_epoch = epoch
            meta = {
                "task": args.task,
                "stage": args.stage,
                "demo_dir": str(demo_dir.resolve()),
                "field_alignment_val": val_align,
                "field_alignment_reach": reach_align,
                "field_alignment_actuate": act_align,
                "best_epoch": best_epoch,
                "image_size": args.image_size,
                "num_train": num_train,
                "num_val": num_val,
                "supervision": args.supervision,
                "input_mode": args.input_mode,
                "in_channels": in_channels,
                "out_dim": args.out_dim,
                "cond_dim": cond_dim,
                "g_phase": args.g_phase,
                "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
                "init_meta": init_meta,
                "config": ExperimentConfig(guidance_mode="anchor_field_demo").to_dict(),
            }
            save_field_checkpoint(output, model, meta=meta)

    print(f"[OK] Best val field_alignment={best_align:.4f} @ epoch {best_epoch}")
    print(f"[OK] Checkpoint: {output}")

    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({"best_val_alignment": best_align, "history": history}, indent=2))
    print(f"[OK] Metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
