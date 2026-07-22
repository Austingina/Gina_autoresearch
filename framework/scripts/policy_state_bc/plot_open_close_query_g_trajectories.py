#!/usr/bin/env python3
"""Plot compact open/close query-G trajectory diagnostics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def grouped_steps(path: Path, policy: str) -> dict[int, list[dict]]:
    rows = json.load(open(path, encoding="utf-8"))
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if row["policy"] == policy:
            groups[int(row["rollout_idx"])].append(row)
    for rollout in groups:
        groups[rollout].sort(key=lambda row: int(row["step"]))
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-metrics", type=Path, required=True)
    parser.add_argument("--targeted-steps", type=Path, required=True)
    parser.add_argument("--targeted-rollouts", type=Path, required=True)
    parser.add_argument("--signflip-aggregates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = json.load(open(args.phase_metrics, encoding="utf-8"))
    rows = [row for row in rows if row["policy"].startswith("no_type_")]
    metrics = [
        ("min_dist_to_part", "Minimum EEF-part distance (m)"),
        ("pre_eef_g_field_cos_mean", "Pre-contact EEF/G-field cosine"),
        ("late_action_desired_axis_mean", "Late desired-axis command"),
        ("max_q_progress_normalized", "Maximum normalized joint progress"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    rng = np.random.default_rng(0)
    for ax, (key, title) in zip(axes.flat, metrics):
        groups = []
        for task in ["open_drawer", "close_drawer"]:
            groups.append([float(row[key]) for row in rows if row["task"] == task and row.get(key) is not None])
        ax.boxplot(groups, labels=["open", "close"], showfliers=False)
        for index, vals in enumerate(groups, start=1):
            x = index + rng.normal(0.0, 0.035, len(vals))
            ax.scatter(x, vals, s=11, alpha=0.35)
        ax.set_title(title)
        ax.grid(alpha=0.2)
    fig.suptitle("No-token-type policies: seen open vs zero-shot close (5 seeds, 25 cases each)")
    fig.tight_layout()
    fig.savefig(args.out_dir / "open_close_distribution.png", dpi=180)
    plt.close(fig)

    close_rows = [row for row in rows if row["task"] == "close_drawer"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in close_rows:
        if row["success"]:
            color, marker, size, label = "#087f5b", "*", 130, "success"
        elif row["effective_contact"]:
            color, marker, size, label = "#e67700", "o", 48, "joint moved, task failed"
        else:
            color, marker, size, label = "#868e96", ".", 25, "no effective contact"
        ax.scatter(
            row["min_dist_to_contact_anchor"],
            row["max_q_progress_normalized"],
            c=color,
            marker=marker,
            s=size,
            alpha=0.8,
            label=label,
        )
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())
    ax.set_xlabel("Minimum EEF-contact-anchor distance (m)")
    ax.set_ylabel("Maximum normalized close progress")
    ax.set_title("Close outcome is gated by contact geometry before push magnitude")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.out_dir / "close_anchor_distance_vs_progress.png", dpi=180)
    plt.close(fig)

    rollout_meta = json.load(open(args.targeted_rollouts, encoding="utf-8"))
    success = {
        int(row["rollout_idx"]): bool(row["success"])
        for row in rollout_meta
        if row["policy"] == "no_type_seed2"
    }
    groups = grouped_steps(args.targeted_steps, "no_type_seed2")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for rollout_idx, trajectory in sorted(groups.items()):
        step = np.asarray([row["step"] for row in trajectory])
        style = "-" if success[rollout_idx] else "--"
        alpha = 0.95 if success[rollout_idx] else 0.55
        label = f"{rollout_idx} {'success' if success[rollout_idx] else 'failure'}"
        axes[0, 0].plot(step, [row["part_to_eef_axis"] for row in trajectory], style, alpha=alpha, label=label)
        axes[0, 1].plot(step, [row["dist_to_contact_anchor"] for row in trajectory], style, alpha=alpha)
        axes[1, 0].plot(step, [row["action_desired_axis"] for row in trajectory], style, alpha=alpha)
        q0 = float(trajectory[0]["q"])
        sign = float(trajectory[0]["desired_sign"])
        joint_range = float(trajectory[0]["joint_range"])
        progress = [sign * (float(row["q_after"]) - q0) / joint_range for row in trajectory]
        axes[1, 1].plot(step, progress, style, alpha=alpha)
    axes[0, 0].axhline(0.0, color="black", linewidth=0.7)
    axes[0, 0].axhline(-0.0627, color="#087f5b", linewidth=1.0, linestyle=":", label="nominal close-side anchor")
    axes[0, 0].set_ylabel("EEF-part offset on close axis (m)")
    axes[0, 1].set_ylabel("EEF-anchor distance (m)")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.7)
    axes[1, 0].set_ylabel("Action projection on close axis")
    axes[1, 1].set_ylabel("Normalized drawer close progress")
    axes[1, 0].set_xlabel("Step")
    axes[1, 1].set_xlabel("Step")
    axes[0, 0].legend(fontsize=8, ncol=2)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle("Seed2 video cases: approach side, anchor distance, command, and drawer progress")
    fig.tight_layout()
    fig.savefig(args.out_dir / "seed2_video_case_trajectories.png", dpi=180)
    plt.close(fig)

    signflip = json.load(open(args.signflip_aggregates, encoding="utf-8"))["by_policy_task"]
    labels = ["seed0 normal", "seed0 flip", "seed2 normal", "seed2 flip"]
    keys = [
        "no_type_seed0_normal::close_drawer",
        "no_type_seed0_sign_flip::close_drawer",
        "no_type_seed2_normal::close_drawer",
        "no_type_seed2_sign_flip::close_drawer",
    ]
    contacts = [signflip[key]["effective_contact_rate"] for key in keys]
    anchor_dist = [signflip[key]["min_dist_to_contact_anchor_p50"] for key in keys]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, contacts, color=["#4c6ef5", "#f59f00", "#4c6ef5", "#f59f00"])
    axes[0].set_ylabel("Effective-contact rate")
    axes[1].bar(labels, anchor_dist, color=["#4c6ef5", "#f59f00", "#4c6ef5", "#f59f00"])
    axes[1].set_ylabel("Median minimum anchor distance (m)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Sign-flip affects approach basin differently across seeds")
    fig.tight_layout()
    fig.savefig(args.out_dir / "normal_vs_signflip_contact.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
