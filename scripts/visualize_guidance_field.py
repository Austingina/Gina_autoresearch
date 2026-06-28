#!/usr/bin/env python3
"""Compare training labels G* vs learned guidance field G_θ on demo masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from goals.field_data import (  # noqa: E402
    build_field_indices_from_demo_dir,
    load_demo_dir,
    read_field_visual_at_index,
    train_val_split,
    train_val_split_indices,
)
from goals.field_model import load_field_checkpoint  # noqa: E402


def mask_to_rgb(mask_chw: np.ndarray, *, view: str = "third_person") -> np.ndarray:
    """Render one camera mask as RGB from stacked visual channels.

    Layouts:
      6ch: ee_mask(0:3) + third_person_mask(3:6)
      8ch: ee_mask(0:3) + ee_depth(3) + third_person_mask(4:7) + tp_depth(7)
    """
    c = mask_chw.shape[0]
    if c >= 8:
        sl = slice(0, 3) if view == "ee" else slice(4, 7)
    elif c >= 6:
        sl = slice(0, 3) if view == "ee" else slice(3, 6)
    else:
        sl = slice(0, min(3, c))
    rgb = np.clip(mask_chw[sl].transpose(1, 2, 0), 0.0, 1.0)
    if rgb.max() < 0.05 and view != "ee" and c >= 6:
        return mask_to_rgb(mask_chw, view="ee")
    return rgb


def draw_arrow(ax, vec: np.ndarray, *, color: str, label: str | None = None, alpha: float = 1.0) -> None:
    """Draw unit direction in robot x-z plane (x→right, z→up)."""
    scale = 0.85
    ax.annotate(
        "",
        xy=(vec[0] * scale, vec[1] * scale),
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5, alpha=alpha, mutation_scale=14),
    )
    if label:
        ax.text(
            vec[0] * scale * 1.15,
            vec[1] * scale * 1.15,
            label,
            color=color,
            fontsize=8,
            ha="center",
            va="center",
            fontweight="bold",
        )


def style_direction_ax(ax, title: str) -> None:
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.axhline(0, color="#cccccc", lw=0.8)
    ax.axvline(0, color="#cccccc", lw=0.8)
    circle = plt.Circle((0, 0), 1.0, fill=False, color="#aaaaaa", ls="--", lw=1.0)
    ax.add_patch(circle)
    ax.set_xlabel("x (robot frame)")
    ax.set_ylabel("z (robot frame)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)


@torch.no_grad()
def predict_all(model, masks: torch.Tensor, device: str, batch_size: int = 256) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(masks), batch_size):
        batch = masks[i : i + batch_size].to(device)
        preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds, axis=0)


def _pick_examples(indices: np.ndarray, alignments: np.ndarray, n: int) -> list[int]:
    order = np.argsort(alignments[indices])
    picks: list[int] = []
    for rel in [order[-1], order[len(order) // 2], order[0]]:
        picks.append(int(indices[rel]))
    remaining = np.linspace(0, len(order) - 1, n, dtype=int)
    for rel in remaining:
        if len(picks) >= n:
            break
        cand = int(indices[order[rel]])
        if cand not in picks:
            picks.append(cand)
    return picks[:n]


def _phase_label(phase: float, supervision: str) -> str:
    if supervision == "subgoal":
        return "pre-grasp EE→cube" if phase < 0.5 else "post-grasp cube→goal"
    return "post-grasp cube→goal"


def _gt_label(supervision: str) -> str:
    if supervision == "subgoal":
        return "G* — subgoal label (phase-aware)"
    if supervision == "mask_flow":
        return "G* — mask centroid flow"
    return "G* — cube→goal (state_goal)"


def main() -> int:
    p = argparse.ArgumentParser(description="Visualize GT vs predicted guidance field")
    p.add_argument("--demo-dir", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--stage", default="s0_fixed", choices=["s0_fixed", "s1_random_cube"])
    p.add_argument("--supervision", default=None, choices=["state_goal", "mask_flow", "subgoal"])
    p.add_argument(
        "--post-grasp-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include only phase>=0.5 samples (default: True for state_goal, False for subgoal)",
    )
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-examples", type=int, default=4, help="Examples per phase row (subgoal mode)")
    p.add_argument("--input-mode", default="mask_depth", choices=["mask", "mask_depth"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--lazy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read val masks from HDF5 on the fly (required for 240px S1).",
    )
    p.add_argument(
        "--max-val",
        type=int,
        default=0,
        help="Cap val samples for faster viz (0 = all).",
    )
    args = p.parse_args()

    if args.supervision is None:
        args.supervision = "subgoal" if args.stage == "s1_random_cube" else "state_goal"
    if args.post_grasp_only is None:
        args.post_grasp_only = args.supervision == "state_goal" and args.stage == "s0_fixed"
    if args.demo_dir is None:
        args.demo_dir = ROOT / f"data/lift/{args.stage}/demos"
    if args.checkpoint is None:
        args.checkpoint = ROOT / f"checkpoints/lift/guidance_field_{args.stage}.pt"
    if args.output is None:
        suffix = "_phases" if args.supervision == "subgoal" and not args.post_grasp_only else ""
        args.output = ROOT / f"outputs/figures/guidance_field_{args.stage}{suffix}.png"

    if not args.checkpoint.is_file():
        print(f"[ERROR] checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1
    if not args.demo_dir.is_dir():
        print(f"[ERROR] demo-dir not found: {args.demo_dir}", file=sys.stderr)
        return 1

    model, meta = load_field_checkpoint(args.checkpoint, device=args.device)
    image_size = int(meta.get("image_size", args.image_size))
    input_mode = str(meta.get("input_mode", args.input_mode))

    if args.lazy:
        indices = build_field_indices_from_demo_dir(
            args.demo_dir,
            supervision=args.supervision,
            post_grasp_only=args.post_grasp_only,
            input_mode=input_mode,
        )
        _, val_idx = train_val_split_indices(indices, val_ratio=args.val_ratio, seed=args.seed)
        if args.max_val and len(val_idx) > args.max_val:
            rng = np.random.default_rng(args.seed)
            val_idx = list(rng.choice(val_idx, size=args.max_val, replace=False))
        files: dict = {}
        val_targets = np.stack([e.target for e in val_idx], axis=0)
        val_phases = np.array([0.0] * len(val_idx), dtype=np.float32)  # filled below
        val_preds = np.empty_like(val_targets)
        val_samples_meta: list = []

        batch_size = 256
        for start in range(0, len(val_idx), batch_size):
            chunk = val_idx[start : start + batch_size]
            visuals = []
            for e in chunk:
                visuals.append(
                    read_field_visual_at_index(e, image_size=image_size, input_mode=input_mode, files=files)
                )
            masks = torch.from_numpy(np.stack(visuals, axis=0))
            val_preds[start : start + len(chunk)] = predict_all(model, masks, args.device, batch_size=len(chunk))
            for e in chunk:
                val_samples_meta.append(e)
        # phase not stored in FieldIndexEntry — read from HDF5 once per unique demo step
        import h5py

        phase_cache: dict[tuple[str, str, int], float] = {}
        for i, e in enumerate(val_idx):
            key = (e.hdf5_path, e.demo, e.step)
            if key not in phase_cache:
                if e.hdf5_path not in files:
                    files[e.hdf5_path] = h5py.File(e.hdf5_path, "r")
                phase_cache[key] = float(np.asarray(files[e.hdf5_path]["data"][e.demo]["goals/phase"][e.step]).reshape(-1)[0])
            val_phases[i] = phase_cache[key]

        class _LazySample:
            def __init__(self, entry, visual: np.ndarray | None = None):
                self.demo = entry.demo
                self.step = entry.step
                self.phase = phase_cache[(entry.hdf5_path, entry.demo, entry.step)]
                self._entry = entry
                self._visual = visual
                self.target = entry.target

            @property
            def visual(self) -> np.ndarray:
                if self._visual is None:
                    self._visual = read_field_visual_at_index(
                        self._entry, image_size=image_size, input_mode=input_mode, files=files
                    )
                return self._visual

        val_samples = [_LazySample(e) for e in val_idx]
    else:
        samples = load_demo_dir(
            args.demo_dir,
            image_size=image_size,
            supervision=args.supervision,
            post_grasp_only=args.post_grasp_only,
            input_mode=input_mode,
        )
        _, val_samples = train_val_split(samples, val_ratio=args.val_ratio, seed=args.seed)
        val_masks = torch.from_numpy(np.stack([s.visual for s in val_samples], axis=0))
        val_targets = np.stack([s.target for s in val_samples], axis=0)
        val_preds = predict_all(model, val_masks, args.device)
        val_phases = np.array([s.phase for s in val_samples], dtype=np.float32)

    alignments = (val_targets * val_preds).sum(axis=1)
    mean_align = float(alignments.mean())

    pre_idx = np.where(val_phases < 0.5)[0]
    post_idx = np.where(val_phases >= 0.5)[0]
    pre_align = float(alignments[pre_idx].mean()) if len(pre_idx) else float("nan")
    post_align = float(alignments[post_idx].mean()) if len(post_idx) else float("nan")

    print(
        f"[INFO] val={len(val_samples)} pre_grasp={len(pre_idx)} post_grasp={len(post_idx)} "
        f"mean_cos={mean_align:.4f} pre={pre_align:.4f} post={post_align:.4f}"
    )

    split_mode = args.supervision == "subgoal" and not args.post_grasp_only and len(pre_idx) and len(post_idx)
    if split_mode:
        pre_picks = _pick_examples(pre_idx, alignments, args.num_examples)
        post_picks = _pick_examples(post_idx, alignments, args.num_examples)
        n_cols = max(len(pre_picks), len(post_picks), 4)
        fig = plt.figure(figsize=(4 * n_cols, 16), facecolor="white")
        gs = GridSpec(7, n_cols, figure=fig, height_ratios=[0.12, 1.2, 1.0, 0.12, 1.2, 1.0, 1.0], hspace=0.5, wspace=0.35)

        fig.suptitle(
            f"S1 subgoal G_θ  |  val cos={mean_align:.3f}  "
            f"pre-grasp={pre_align:.3f}  post-grasp={post_align:.3f}  "
            f"epoch={meta.get('best_epoch', '?')}",
            fontsize=13,
            fontweight="bold",
            y=0.99,
        )

        for row_label, picks, hdr_row, mask_row, dir_row, cam_view in [
            ("PRE-GRASP (G* = EE → cube)", pre_picks, 0, 1, 2, "ee"),
            ("POST-GRASP (G* = cube → goal)", post_picks, 3, 4, 5, "third_person"),
        ]:
            ax_hdr = fig.add_subplot(gs[hdr_row, :])
            ax_hdr.axis("off")
            ax_hdr.text(0.5, 0.5, row_label, ha="center", va="center", fontsize=12, fontweight="bold")

            for col, idx in enumerate(picks):
                s = val_samples[idx]
                gt = val_targets[idx]
                pred = val_preds[idx]
                cos = float(alignments[idx])

                ax_mask = fig.add_subplot(gs[mask_row, col])
                if cam_view == "ee":
                    ee_rgb = mask_to_rgb(s.visual, view="ee")
                    tp_rgb = mask_to_rgb(s.visual, view="third_person")
                    panel = np.concatenate([ee_rgb, tp_rgb], axis=1)
                    ax_mask.imshow(panel)
                    ax_mask.set_title(
                        f"{s.demo} step{s.step}\nL=ee_mask R=third_person | cos={cos:.3f}",
                        fontsize=8,
                    )
                else:
                    ax_mask.imshow(mask_to_rgb(s.visual, view=cam_view))
                    ax_mask.set_title(
                        f"{s.demo} step{s.step} [{cam_view}]\nphase={s.phase:.2f} cos={cos:.3f}",
                        fontsize=8,
                    )
                ax_mask.axis("off")

                ax_dir = fig.add_subplot(gs[dir_row, col])
                style_direction_ax(ax_dir, _phase_label(s.phase, args.supervision))
                draw_arrow(ax_dir, gt, color="#2ca02c", label="G*")
                draw_arrow(ax_dir, pred, color="#d62728", label="Gθ")

        ax_hist = fig.add_subplot(gs[6, : n_cols // 2])
        if len(pre_idx):
            ax_hist.hist(alignments[pre_idx], bins=35, alpha=0.65, color="#1f77b4", label=f"pre-grasp (n={len(pre_idx)})")
        if len(post_idx):
            ax_hist.hist(alignments[post_idx], bins=35, alpha=0.65, color="#ff7f0e", label=f"post-grasp (n={len(post_idx)})")
        ax_hist.axvline(pre_align, color="#1f77b4", ls="--", lw=2)
        ax_hist.axvline(post_align, color="#ff7f0e", ls="--", lw=2)
        ax_hist.axvline(0.75, color="gray", ls=":", lw=1.5, label="threshold 0.75")
        ax_hist.set_xlabel("cosine(G*, Gθ)")
        ax_hist.set_ylabel("count")
        ax_hist.set_title("Phase-wise alignment")
        ax_hist.legend(fontsize=8)
        ax_hist.grid(True, alpha=0.25)

        ax_scatter = fig.add_subplot(gs[6, n_cols // 2 :])
        style_direction_ax(ax_scatter, "Val directions (subsample)")
        rng = np.random.default_rng(args.seed)
        sub_idx = rng.choice(len(val_samples), size=min(600, len(val_samples)), replace=False)
        for i in sub_idx:
            if val_phases[i] < 0.5:
                ax_scatter.plot([val_targets[i, 0], val_preds[i, 0]], [val_targets[i, 1], val_preds[i, 1]], color="#1f77b4", alpha=0.08, lw=0.5)
            else:
                ax_scatter.plot([val_targets[i, 0], val_preds[i, 0]], [val_targets[i, 1], val_preds[i, 1]], color="#ff7f0e", alpha=0.08, lw=0.5)
        ax_scatter.scatter(val_targets[sub_idx, 0], val_targets[sub_idx, 1], s=10, c="#2ca02c", alpha=0.4, label="G*")
        ax_scatter.scatter(val_preds[sub_idx, 0], val_preds[sub_idx, 1], s=10, c="#d62728", alpha=0.4, label="Gθ")
        ax_scatter.legend(loc="upper left", fontsize=8)
    else:
        picks = _pick_examples(np.arange(len(val_samples)), alignments, args.num_examples)
        n_ex = len(picks)
        fig = plt.figure(figsize=(16, 10), facecolor="white")
        gs = GridSpec(3, max(n_ex, 4), figure=fig, height_ratios=[1.2, 1.0, 1.1], hspace=0.45, wspace=0.35)
        fig.suptitle(
            f"Guidance field G* vs G_θ  |  val cos={mean_align:.4f}  "
            f"supervision={args.supervision}  epoch={meta.get('best_epoch', '?')}",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )
        for col, idx in enumerate(picks):
            s = val_samples[idx]
            ax_mask = fig.add_subplot(gs[0, col])
            ax_mask.imshow(mask_to_rgb(s.visual, view="third_person"))
            ax_mask.set_title(f"{s.demo}\nstep {s.step}  cos={alignments[idx]:.3f}", fontsize=8)
            ax_mask.axis("off")
            ax_dir = fig.add_subplot(gs[1, col])
            style_direction_ax(ax_dir, _phase_label(s.phase, args.supervision))
            draw_arrow(ax_dir, val_targets[idx], color="#2ca02c", label="G*")
            draw_arrow(ax_dir, val_preds[idx], color="#d62728", label="Gθ")

        ax_scatter = fig.add_subplot(gs[2, : max(n_ex // 2, 2)])
        style_direction_ax(ax_scatter, f"All val (n={len(val_samples)})")
        rng = np.random.default_rng(args.seed)
        sub_idx = rng.choice(len(val_samples), size=min(800, len(val_samples)), replace=False)
        for i in sub_idx:
            ax_scatter.plot([val_targets[i, 0], val_preds[i, 0]], [val_targets[i, 1], val_preds[i, 1]], color="#999999", alpha=0.15, lw=0.6)
        ax_scatter.scatter(val_targets[sub_idx, 0], val_targets[sub_idx, 1], s=12, c="#2ca02c", alpha=0.55, label="G*")
        ax_scatter.scatter(val_preds[sub_idx, 0], val_preds[sub_idx, 1], s=12, c="#d62728", alpha=0.55, label="Gθ")
        ax_scatter.legend(loc="upper left", fontsize=9)

        ax_hist = fig.add_subplot(gs[2, max(n_ex // 2, 2) :])
        ax_hist.hist(alignments, bins=40, color="#1f77b4", alpha=0.85, edgecolor="white")
        ax_hist.axvline(mean_align, color="#d62728", ls="--", lw=2, label=f"mean={mean_align:.4f}")
        ax_hist.axvline(0.75, color="#ff7f0e", ls=":", lw=2, label="threshold 0.75")
        ax_hist.set_xlabel("cosine(G*, Gθ)")
        ax_hist.set_ylabel("count")
        ax_hist.set_title("Validation alignment")
        ax_hist.legend(fontsize=9)
        ax_hist.grid(True, alpha=0.25)

    legend_handles = [
        mpatches.Patch(color="#2ca02c", label=_gt_label(args.supervision)),
        mpatches.Patch(color="#d62728", label="Gθ — MaskFieldCNN (mask+depth, no phase input)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.01))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[OK] Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
