# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot step reward and success-episode distance histograms from offline CQL HDF5."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

_OFFLINE_CQL_ROOT = (
    Path(__file__).resolve().parents[2]
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/mdp/teacher_student"
)
if str(_OFFLINE_CQL_ROOT) not in sys.path:
    sys.path.insert(0, str(_OFFLINE_CQL_ROOT))

from offline_cql.rewards import compute_offline_reach_rewards  # noqa: E402

DISTANCE_THRESHOLDS_CM = (1.0, 2.0, 3.0, 10.0)


def _demo_sort_key(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else 0


def _read_demo_reward_parts(group: h5py.Group) -> dict[str, np.ndarray]:
    ee_pose = np.asarray(group["ee_pose"], dtype=np.float32)
    student_state = np.asarray(group["obs/student_state"], dtype=np.float32)
    success_score = np.asarray(group["rewards/success_score"], dtype=np.float32).reshape(-1)
    num_steps = min(len(ee_pose), len(student_state), len(success_score))
    return compute_offline_reach_rewards(
        ee_pose[:num_steps, :3],
        student_state[:num_steps, 4:7],
        success_score=success_score[:num_steps],
    )


def load_training_rewards(dataset_path: str) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with h5py.File(dataset_path, "r") as h5:
        demo_names = sorted(h5["data"].keys(), key=_demo_sort_key)
        for demo_name in demo_names:
            chunks.append(_read_demo_reward_parts(h5["data"][demo_name])["rewards"])
    if not chunks:
        raise ValueError(f"No demos found in {dataset_path}")
    return np.concatenate(chunks, axis=0)


def load_success_episode_distances(dataset_path: str) -> tuple[np.ndarray, dict[str, int]]:
    """Return ee_cube_distance (meters) for all steps in episode-success demos."""
    chunks: list[np.ndarray] = []
    num_success = 0
    num_total = 0
    with h5py.File(dataset_path, "r") as h5:
        demo_names = sorted(h5["data"].keys(), key=_demo_sort_key)
        for demo_name in demo_names:
            group = h5["data"][demo_name]
            num_total += 1
            if not bool(group.attrs.get("success", True)):
                continue
            num_success += 1
            chunks.append(_read_demo_reward_parts(group)["ee_cube_distance"])
    if not chunks:
        raise ValueError(f"No successful demos found in {dataset_path}")
    distances = np.concatenate(chunks, axis=0)
    return distances, {
        "num_success_episodes": num_success,
        "num_total_episodes": num_total,
        "num_steps": int(distances.size),
    }


def summarize_distance_thresholds(distances_m: np.ndarray, thresholds_cm: tuple[float, ...]) -> dict[float, dict[str, float]]:
    """Count and fraction of steps with ee_cube_distance below each threshold."""
    total = int(distances_m.size)
    stats: dict[float, dict[str, float]] = {}
    for cm in thresholds_cm:
        thr_m = cm / 100.0
        count = int(np.sum(distances_m < thr_m))
        stats[cm] = {
            "count": count,
            "fraction": float(count / total) if total > 0 else 0.0,
        }
    return stats


def print_success_distance_stats(
    distances_m: np.ndarray,
    *,
    dataset_path: str,
    episode_stats: dict[str, int],
    thresholds_cm: tuple[float, ...] = DISTANCE_THRESHOLDS_CM,
) -> dict[float, dict[str, float]]:
    distances_cm = distances_m * 100.0
    threshold_stats = summarize_distance_thresholds(distances_m, thresholds_cm)

    print(
        f"[INFO] success-episode ee_cube_distance: "
        f"{episode_stats['num_success_episodes']}/{episode_stats['num_total_episodes']} episodes, "
        f"{episode_stats['num_steps']:,} steps"
    )
    print(
        f"  min={distances_cm.min():.3f} cm  max={distances_cm.max():.3f} cm  "
        f"mean={distances_cm.mean():.3f} cm  median={np.median(distances_cm):.3f} cm"
    )
    print(
        f"  p50={np.percentile(distances_cm, 50):.3f} cm  "
        f"p95={np.percentile(distances_cm, 95):.3f} cm  "
        f"p99={np.percentile(distances_cm, 99):.3f} cm"
    )
    print(f"  threshold coverage ({Path(dataset_path).name}):")
    for cm in thresholds_cm:
        entry = threshold_stats[cm]
        print(f"    < {cm:g} cm: {entry['count']:,} steps ({entry['fraction'] * 100:.2f}%)")
    return threshold_stats


def sample_rewards(rewards: np.ndarray, num_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if rewards.size >= num_samples:
        idx = rng.choice(rewards.size, size=num_samples, replace=False)
        return rewards[idx]
    if rewards.size == 0:
        raise ValueError("Reward array is empty.")
    idx = rng.choice(rewards.size, size=num_samples, replace=True)
    return rewards[idx]


def resolve_xlim(
    rewards: np.ndarray,
    xlim: tuple[float, float] | None,
    percentile_low: float,
    percentile_high: float,
    pad_frac: float,
) -> tuple[float, float]:
    if xlim is not None:
        return xlim
    lo = float(np.percentile(rewards, percentile_low))
    hi = float(np.percentile(rewards, percentile_high))
    pad = max((hi - lo) * pad_frac, 1e-4)
    return lo - pad, hi + pad


def plot_histogram(
    rewards: np.ndarray,
    *,
    output_path: Path,
    dataset_path: str,
    num_requested: int,
    num_total: int,
    bins: int,
    xlim: tuple[float, float] | None,
    percentile_low: float,
    percentile_high: float,
    pad_frac: float,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    x_min, x_max = resolve_xlim(rewards, xlim, percentile_low, percentile_high, pad_frac)
    in_range = (rewards >= x_min) & (rewards <= x_max)
    clipped_frac = 1.0 - float(np.mean(in_range))

    counts, edges, patches = ax.hist(
        rewards,
        bins=bins,
        range=(x_min, x_max),
        color="#4C72B0",
        edgecolor="white",
        alpha=0.9,
    )
    ax.set_xlim(x_min, x_max)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, label="zero")
    ax.axvline(float(np.mean(rewards)), color="#C44E52", linestyle="-", linewidth=1.5, label=f"mean={np.mean(rewards):.4f}")
    ax.axvline(float(np.median(rewards)), color="#55A868", linestyle="-", linewidth=1.5, label=f"median={np.median(rewards):.4f}")

    sampled_note = (
        f"sampled={rewards.size:,} (all available)"
        if rewards.size == num_total and num_total < num_requested
        else f"sampled={rewards.size:,} / total={num_total:,}"
    )
    ax.set_title(
        f"Step reward distribution (reach_distance reward_total)\n"
        f"{Path(dataset_path).name} | {sampled_note}\n"
        f"x in [{x_min:.3f}, {x_max:.3f}] ({percentile_low:g}–{percentile_high:g} pct + pad; "
        f"{clipped_frac * 100:.2f}% outliers excluded)",
        fontsize=11,
    )
    ax.set_xlabel("reward_total (reach + near + success_bonus)")
    ax.set_ylabel("count")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)

    stats_text = (
        f"min={np.min(rewards):.4f}\n"
        f"max={np.max(rewards):.4f}\n"
        f"std={np.std(rewards):.4f}\n"
        f"p01={np.percentile(rewards, 1):.4f}\n"
        f"p99={np.percentile(rewards, 99):.4f}\n"
        f"positive={np.mean(rewards > 0):.3f}\n"
        f"negative={np.mean(rewards < 0):.3f}\n"
        f"zero={np.mean(rewards == 0):.3f}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_success_distance_histogram(
    distances_m: np.ndarray,
    *,
    output_path: Path,
    dataset_path: str,
    episode_stats: dict[str, int],
    threshold_stats: dict[float, dict[str, float]],
    bins: int,
    xlim_cm: tuple[float, float] | None,
    thresholds_cm: tuple[float, ...] = DISTANCE_THRESHOLDS_CM,
) -> None:
    distances_cm = distances_m * 100.0
    fig, ax = plt.subplots(figsize=(10, 6))

    if xlim_cm is None:
        x_min, x_max = 0.0, max(float(np.percentile(distances_cm, 99.5)), thresholds_cm[-1] * 1.2)
    else:
        x_min, x_max = xlim_cm

    ax.hist(
        distances_cm,
        bins=bins,
        range=(x_min, x_max),
        color="#8172B3",
        edgecolor="white",
        alpha=0.9,
    )
    ax.set_xlim(x_min, x_max)
    ax.axvline(float(np.mean(distances_cm)), color="#C44E52", linestyle="-", linewidth=1.5, label=f"mean={np.mean(distances_cm):.2f} cm")
    ax.axvline(float(np.median(distances_cm)), color="#55A868", linestyle="-", linewidth=1.5, label=f"median={np.median(distances_cm):.2f} cm")

    threshold_colors = ("#DD8452", "#937860", "#DA8BC3", "#4C72B0")
    for cm, color in zip(thresholds_cm, threshold_colors, strict=True):
        ax.axvline(cm, color=color, linestyle="--", linewidth=1.0, alpha=0.85, label=f"< {cm:g} cm")

    ax.set_title(
        f"ee_cube_distance in successful episodes\n"
        f"{Path(dataset_path).name} | "
        f"{episode_stats['num_success_episodes']}/{episode_stats['num_total_episodes']} episodes | "
        f"{episode_stats['num_steps']:,} steps\n"
        f"x in [{x_min:.2f}, {x_max:.2f}] cm",
        fontsize=11,
    )
    ax.set_xlabel("ee_cube_distance (cm)")
    ax.set_ylabel("count")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)

    stats_text = (
        f"min={distances_cm.min():.3f} cm\n"
        f"max={distances_cm.max():.3f} cm\n"
        f"std={distances_cm.std():.3f} cm\n"
        f"p95={np.percentile(distances_cm, 95):.3f} cm\n"
        f"p99={np.percentile(distances_cm, 99):.3f} cm\n"
    )
    for cm in thresholds_cm:
        entry = threshold_stats[cm]
        stats_text += f"<{cm:g}cm={entry['fraction'] * 100:.2f}%\n"

    ax.text(
        0.02,
        0.98,
        stats_text.rstrip(),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reach-distance reward_total histogram from offline CQL HDF5.")
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="./datasets/lift_cube_lenient_teacher_maskdepth/lift_cube_lenient_teacher_maskdepth_v4.hdf5",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1_000_000,
        help="Random transitions to plot. Uses all transitions if dataset is smaller.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="Fixed x-axis limits. Default: auto from percentiles.",
    )
    parser.add_argument(
        "--xlim_percentile",
        type=float,
        nargs=2,
        default=(1.0, 99.0),
        metavar=("LOW", "HIGH"),
        help="Percentile range for auto x-axis limits.",
    )
    parser.add_argument(
        "--xlim_pad_frac",
        type=float,
        default=0.15,
        help="Padding fraction added on both sides of the percentile range.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output PNG path. Defaults to <dataset_stem>_reward_hist.png beside dataset.",
    )
    parser.add_argument(
        "--distance_output",
        type=str,
        default="",
        help="Output PNG for success-episode ee_cube_distance histogram. "
        "Defaults to <dataset_stem>_success_distance_hist.png beside dataset.",
    )
    parser.add_argument(
        "--distance_xlim_cm",
        type=float,
        nargs=2,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="Fixed x-axis limits (cm) for the distance histogram.",
    )
    parser.add_argument(
        "--distance_bins",
        type=int,
        default=80,
        help="Histogram bins for success-episode ee_cube_distance plot.",
    )
    parser.add_argument(
        "--skip_distance",
        action="store_true",
        help="Skip success-episode ee_cube_distance stats and histogram.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_file)
    all_rewards = load_training_rewards(args.dataset_file)
    sampled = sample_rewards(all_rewards, args.num_samples, args.seed)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = dataset_path.with_name(f"{dataset_path.stem}_reward_hist.png")

    plot_histogram(
        sampled,
        output_path=output_path,
        dataset_path=args.dataset_file,
        num_requested=args.num_samples,
        num_total=int(all_rewards.size),
        bins=args.bins,
        xlim=tuple(args.xlim) if args.xlim is not None else None,
        percentile_low=float(args.xlim_percentile[0]),
        percentile_high=float(args.xlim_percentile[1]),
        pad_frac=float(args.xlim_pad_frac),
    )

    print(f"[INFO] total transitions in dataset: {all_rewards.size:,}")
    if all_rewards.size < args.num_samples:
        print(
            f"[INFO] dataset has fewer than {args.num_samples:,} transitions; "
            f"plotted all {all_rewards.size:,} (no oversampling)."
        )
    else:
        print(f"[INFO] randomly sampled {sampled.size:,} transitions (seed={args.seed})")
    print(f"[INFO] saved histogram to: {output_path.resolve()}")

    if not args.skip_distance:
        success_distances, episode_stats = load_success_episode_distances(args.dataset_file)
        threshold_stats = print_success_distance_stats(
            success_distances,
            dataset_path=args.dataset_file,
            episode_stats=episode_stats,
        )

        if args.distance_output:
            distance_output_path = Path(args.distance_output)
        else:
            distance_output_path = dataset_path.with_name(f"{dataset_path.stem}_success_distance_hist.png")

        plot_success_distance_histogram(
            success_distances,
            output_path=distance_output_path,
            dataset_path=args.dataset_file,
            episode_stats=episode_stats,
            threshold_stats=threshold_stats,
            bins=args.distance_bins,
            xlim_cm=tuple(args.distance_xlim_cm) if args.distance_xlim_cm is not None else None,
        )
        print(f"[INFO] saved success-episode distance histogram to: {distance_output_path.resolve()}")


if __name__ == "__main__":
    main()
