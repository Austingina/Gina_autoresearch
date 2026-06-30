#!/usr/bin/env python3
"""Delete or truncate oversized DAgger disk episodes that stall replay sampling."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import torch


def main() -> int:
    p = argparse.ArgumentParser(description="Prune oversized vision DAgger disk episodes")
    p.add_argument("--log-dir", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=512)
    p.add_argument("--max-file-mb", type=float, default=400.0, help="Fast pre-filter before loading tensors")
    args = p.parse_args()

    episodes_dir = args.log_dir / "dagger_episodes" / "episodes"
    if not episodes_dir.is_dir():
        raise SystemExit(f"[ERROR] Missing {episodes_dir}")

    deleted = 0
    kept = 0
    kept_steps = 0
    for path in sorted(glob.glob(str(episodes_dir / "ep_*.pt"))):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb > args.max_file_mb:
            os.remove(path)
            deleted += 1
            print(f"[DEL] {os.path.basename(path)} size={size_mb:.0f}MB (> {args.max_file_mb}MB)")
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        visual = payload.get("visual")
        if visual is None:
            os.remove(path)
            deleted += 1
            print(f"[DEL] {os.path.basename(path)} missing visual")
            continue
        length = int(visual.shape[0])
        if length > args.max_steps:
            os.remove(path)
            deleted += 1
            print(f"[DEL] {os.path.basename(path)} len={length} (> {args.max_steps})")
            continue
        kept += 1
        kept_steps += length

    print(f"[OK] kept={kept} deleted={deleted} total_steps={kept_steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
