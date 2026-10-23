#!/usr/bin/env python3
"""Download and hash-verify pinned RoboTwin official demo_clean archives."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import threading
from pathlib import Path

SAFE_TASK = re.compile(r"^[A-Za-z0-9_.-]+$")
PRINT_LOCK = threading.Lock()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(row: dict, *, repo: str, revision: str, output_root: Path, backend: str) -> dict:
    task = str(row["task"])
    if not SAFE_TASK.fullmatch(task):
        raise ValueError(f"unsafe task name: {task!r}")
    remote_path = str(row["path"])
    expected_bytes = int(row["bytes"])
    expected_sha = str(row["lfs"]["oid"])
    destination = output_root / remote_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    if destination.is_file():
        actual_sha = sha256_file(destination)
        if destination.stat().st_size == expected_bytes and actual_sha == expected_sha:
            with PRINT_LOCK:
                print(f"[verified-existing] {task} bytes={expected_bytes}", flush=True)
            return {"task": task, "path": str(destination), "bytes": expected_bytes, "sha256": actual_sha, "status": "verified-existing"}
        raise ValueError(f"existing destination mismatches manifest: {destination}")

    if backend == "hf":
        command = [
            "hf", "download", repo, remote_path, "--repo-type", "dataset",
            "--revision", revision, "--local-dir", os.fspath(output_root),
            "--format", "quiet",
        ]
        subprocess.run(command, check=True)
        candidate = destination
    else:
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{remote_path}"
        command = [
            "curl", "--fail", "--location", "--silent", "--show-error",
            "--retry", "8", "--retry-all-errors", "--retry-delay", "5",
            "--continue-at", "-", "--output", os.fspath(partial), url,
        ]
        subprocess.run(command, check=True)
        candidate = partial
    if candidate.stat().st_size != expected_bytes:
        raise ValueError(f"size mismatch for {task}: {candidate.stat().st_size} != {expected_bytes}")
    actual_sha = sha256_file(candidate)
    if actual_sha != expected_sha:
        raise ValueError(f"SHA-256 mismatch for {task}: {actual_sha} != {expected_sha}")
    if candidate == partial:
        partial.replace(destination)
    with PRINT_LOCK:
        print(f"[downloaded+verified] {task} bytes={expected_bytes}", flush=True)
    return {"task": task, "path": str(destination), "bytes": expected_bytes, "sha256": actual_sha, "status": "downloaded+verified"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--backend", choices=("hf", "curl"), default="hf")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")

    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    repo = str(inventory["repo"])
    revision = str(inventory["source_revision"])
    rows = list(inventory["files"])
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.workers, len(rows))) as pool:
        futures = {
            pool.submit(download_one, row, repo=repo, revision=revision, output_root=output_root, backend=args.backend): row["task"]
            for row in rows
        }
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                results.append(future.result())
            except BaseException as exc:
                failures.append({"task": task, "error": f"{type(exc).__name__}: {exc}"})
                with PRINT_LOCK:
                    print(f"[failed] {task}: {exc}", flush=True)

    results.sort(key=lambda row: row["task"])
    receipt = {
        "schema_version": "robotwin_official_download_receipt_v1",
        "repo": repo,
        "revision": revision,
        "provenance_class": "official_trajectory_reorganization_and_label_enrichment",
        "archive_count": len(results),
        "verified_bytes": sum(int(row["bytes"]) for row in results),
        "failures": failures,
        "files": results,
        "complete": not failures and len(results) == len(rows),
    }
    receipt_path = output_root.parent / "download_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in ("archive_count", "verified_bytes", "failures", "complete")}, indent=2), flush=True)
    return 0 if receipt["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
