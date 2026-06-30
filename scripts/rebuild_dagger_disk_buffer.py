#!/usr/bin/env python3
"""Rebuild dagger_aggregate.pt index for a disk-backed vision DAgger buffer."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import torch


def main() -> int:
    p = argparse.ArgumentParser(description="Rebuild dagger_aggregate.pt from disk episodes")
    p.add_argument("--log-dir", type=Path, required=True, help="Run dir containing dagger_episodes/")
    p.add_argument("--max-episodes", type=int, default=250)
    p.add_argument("--success-reserve-ratio", type=float, default=0.35)
    p.add_argument("--visual-storage-dtype", default="float16")
    p.add_argument("--success-count", type=int, default=0, help="Override: mark N longest eps as teacher_success")
    p.add_argument("--max-episode-steps", type=int, default=512, help="Skip episodes longer than this when rebuilding")
    p.add_argument("--max-file-mb", type=float, default=400.0, help="Skip episode files larger than this")
    args = p.parse_args()

    storage_dir = args.log_dir.resolve() / "dagger_episodes"
    episodes_dir = storage_dir / "episodes"
    if not episodes_dir.is_dir():
        raise SystemExit(f"[ERROR] Missing {episodes_dir}")

    entries: list[dict] = []
    max_counter = 0
    for path in sorted(glob.glob(str(episodes_dir / "ep_*.pt"))):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb > args.max_file_mb:
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        visual = payload.get("visual")
        if visual is None:
            continue
        length = int(visual.shape[0])
        if length > args.max_episode_steps:
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            max_counter = max(max_counter, int(stem.split("_")[-1]))
        except ValueError:
            pass
        entries.append(
            {
                "path": os.path.join("episodes", os.path.basename(path)),
                "length": length,
                "has_teacher_phase": payload.get("teacher_phase") is not None,
                "has_proprio": payload.get("proprio") is not None,
            }
        )

    if not entries:
        raise SystemExit(f"[ERROR] No episodes under {episodes_dir}")

    entries.sort(key=lambda e: e["length"], reverse=True)
    n_success = args.success_count if args.success_count > 0 else max(1, int(round(len(entries) * args.success_reserve_ratio)))
    n_success = min(n_success, len(entries))

    success_eps = []
    rolling_eps = []
    for i, entry in enumerate(entries):
        meta = {**entry, "teacher_success": i < n_success}
        if i < n_success:
            success_eps.append(meta)
        else:
            rolling_eps.append(meta)

    total_steps = sum(e["length"] for e in entries)
    state = {
        "backend": "disk",
        "max_episodes": args.max_episodes,
        "success_reserve_ratio": args.success_reserve_ratio,
        "total_steps": total_steps,
        "storage_dir": str(storage_dir),
        "visual_storage_dtype": args.visual_storage_dtype,
        "episode_counter": max_counter,
        "max_store_episode_steps": args.max_episode_steps,
        "success_episodes": success_eps,
        "rolling_episodes": rolling_eps,
    }
    out = args.log_dir / "dagger_aggregate.pt"
    torch.save(state, out)
    print(
        f"[OK] {out} episodes={len(entries)} success={len(success_eps)} rolling={len(rolling_eps)} "
        f"steps={total_steps} counter={max_counter}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
