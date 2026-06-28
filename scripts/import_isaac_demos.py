#!/usr/bin/env python3
"""
Copy demonstration HDF5 files from IsaacLab into AutoResearch-owned storage.

IsaacLab datasets/ is READ-ONLY for this project. Import creates a manifest
with source path + sha256 so copies stay traceable without mutating upstream data.

Usage:
  python scripts/import_isaac_demos.py \\
    --source $ISAACLAB_ROOT/datasets/lift_cube_lenient_teacher_maskdepth_smoke.hdf5 \\
    --task lift --stage s0_fixed

  python scripts/import_isaac_demos.py \\
    --source $ISAACLAB_ROOT/datasets/lift_cube_lenient_teacher_maskdepth \\
    --task lift --stage s0_fixed --glob '*.hdf5'
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from goals.paths import (  # noqa: E402
    ensure_workspace_dirs,
    manifests_dir,
    task_data_dir,
)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _collect_sources(source: Path, glob_pat: str) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        files = sorted(source.glob(glob_pat))
        return [p for p in files if p.is_file()]
    raise FileNotFoundError(f"Source not found: {source}")


def main() -> int:
    p = argparse.ArgumentParser(description="Import Isaac HDF5 demos into AutoResearch data/")
    p.add_argument("--source", type=Path, required=True, help="Isaac .hdf5 file or directory")
    p.add_argument("--task", default="lift")
    p.add_argument("--stage", default="s0_fixed")
    p.add_argument("--glob", default="*.hdf5", help="When source is a directory")
    p.add_argument("--dest", type=Path, default=None, help="Override destination dir")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    ensure_workspace_dirs(args.task, args.stage)
    dest_dir = args.dest or task_data_dir(args.task, args.stage)
    dest_dir.mkdir(parents=True, exist_ok=True)

    sources = _collect_sources(args.source, args.glob)
    if not sources:
        print(f"[ERROR] No files matched under {args.source}", file=sys.stderr)
        return 1

    manifest_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "import_id": manifest_id,
        "task": args.task,
        "stage": args.stage,
        "source_root": str(args.source.resolve()),
        "dest_dir": str(dest_dir.resolve()),
        "files": [],
    }

    for src in sources:
        dst = dest_dir / src.name
        digest = _sha256(src)
        entry = {
            "source": str(src.resolve()),
            "dest": str(dst.resolve()),
            "sha256": digest,
            "size_bytes": src.stat().st_size,
        }
        manifest["files"].append(entry)

        if args.dry_run:
            print(f"[dry-run] {src.name} -> {dst}")
            continue

        if dst.exists():
            if _sha256(dst) == digest:
                print(f"[skip] {dst.name} (identical copy exists)")
                continue
            stem, suffix = dst.stem, dst.suffix
            dst = dest_dir / f"{stem}_{manifest_id}{suffix}"
            entry["dest"] = str(dst.resolve())
            print(f"[warn] Name collision — writing {dst.name}")

        print(f"[copy] {src.name} -> {dst}")
        shutil.copy2(src, dst)

    manifest_path = manifests_dir() / f"import_{args.task}_{args.stage}_{manifest_id}.json"
    if not args.dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"[OK] Manifest: {manifest_path}")
        print(f"[OK] {len(manifest['files'])} file(s) in {dest_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
