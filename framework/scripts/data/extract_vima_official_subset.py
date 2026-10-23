#!/usr/bin/env python3
"""Select and CRC-verify a deterministic VIMA official-data subset via HTTP ranges."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import struct
import threading
import time
from typing import Any, Iterable, Sequence
from zipfile import ZipFile, ZipInfo
import zlib

import requests

DEFAULT_TASKS = (
    "rotate",
    "twist",
    "sweep_without_exceeding",
    "follow_order",
    "pick_in_order_then_restore",
    "rearrange_then_restore",
)
ARCHIVE_PREFIX = "vima_v6/"
TRAJECTORY_MEMBER = re.compile(r"^vima_v6/([^/]+)/(\d{6})/(.+)$")
LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
LOCAL_FILE_SIGNATURE = 0x04034B50


def uniform_sample_ids(ids: Sequence[str], count: int) -> list[str]:
    ordered = sorted(set(ids))
    if count < 1:
        raise ValueError("count must be positive")
    if len(ordered) < count:
        raise ValueError(f"only {len(ordered)} native trajectories available for {count} requested")
    return [ordered[((2 * index + 1) * len(ordered)) // (2 * count)] for index in range(count)]


def inventory_selected_tasks(
    archive: ZipFile, tasks: Sequence[str]
) -> dict[str, dict[str, tuple[str, ...]]]:
    wanted = set(tasks)
    grouped: dict[str, dict[str, list[str]]] = {
        task: defaultdict(list) for task in tasks
    }
    for info in archive.infolist():
        if info.is_dir():
            continue
        match = TRAJECTORY_MEMBER.match(info.filename)
        if not match:
            continue
        task, trajectory_id, _relative = match.groups()
        if task in wanted:
            grouped[task][trajectory_id].append(info.filename)
    missing = [task for task in tasks if not grouped[task]]
    if missing:
        raise ValueError(f"official ZIP has no trajectories for tasks: {missing}")
    return {
        task: {
            trajectory_id: tuple(sorted(names))
            for trajectory_id, names in trajectories.items()
        }
        for task, trajectories in grouped.items()
    }


def validate_native_member_set(names: Iterable[str]) -> None:
    names = tuple(names)
    relative = [name.split("/", 3)[-1] for name in names]
    required = {"action.pkl", "obs.pkl", "trajectory.pkl"}
    missing = sorted(required - set(relative))
    if missing:
        raise ValueError(f"missing required native members: {missing}")
    views: dict[str, set[int]] = {}
    for view in ("rgb_front", "rgb_top"):
        frames = {
            int(PurePosixPath(name).stem)
            for name in relative
            if name.startswith(f"{view}/") and name.endswith(".jpg")
        }
        if not frames:
            raise ValueError(f"missing required native members: {view}")
        views[view] = frames
    if views["rgb_front"] != views["rgb_top"]:
        raise ValueError("rgb_front/rgb_top frame IDs differ")
    expected = set(range(max(views["rgb_front"]) + 1))
    if views["rgb_front"] != expected:
        raise ValueError("RGB frame IDs are not contiguous from zero")


def _member_end(info: ZipInfo) -> int:
    return (
        int(info.header_offset)
        + LOCAL_HEADER.size
        + len(info.filename.encode("utf-8"))
        + len(info.extra)
        + int(info.compress_size)
        + 128
    )


def _member_bytes(payload: bytes, range_start: int, info: ZipInfo) -> bytes:
    offset = int(info.header_offset) - range_start
    header = LOCAL_HEADER.unpack_from(payload, offset)
    signature, _version, _flags, method, _mtime, _mdate, _crc, _csize, _usize, name_len, extra_len = header
    if signature != LOCAL_FILE_SIGNATURE:
        raise RuntimeError(f"invalid local header: {info.filename}")
    data_start = offset + LOCAL_HEADER.size + name_len + extra_len
    compressed = payload[data_start : data_start + int(info.compress_size)]
    if len(compressed) != int(info.compress_size):
        raise RuntimeError(f"truncated compressed member: {info.filename}")
    if method == 0:
        data = compressed
    elif method == 8:
        data = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise RuntimeError(f"unsupported ZIP method {method}: {info.filename}")
    if len(data) != int(info.file_size):
        raise RuntimeError(f"size mismatch: {info.filename}")
    if zlib.crc32(data) & 0xFFFFFFFF != int(info.CRC):
        raise RuntimeError(f"CRC mismatch: {info.filename}")
    return data


def _local_destination(output_root: Path, filename: str) -> Path:
    path = PurePosixPath(filename)
    if not filename.startswith(ARCHIVE_PREFIX) or ".." in path.parts:
        raise ValueError(f"unsafe archive member: {filename}")
    destination = output_root.joinpath(*path.parts[1:])
    destination.resolve().relative_to(output_root.resolve())
    return destination


def _existing_valid(path: Path, info: ZipInfo) -> bool:
    if not path.is_file() or path.stat().st_size != int(info.file_size):
        return False
    return zlib.crc32(path.read_bytes()) & 0xFFFFFFFF == int(info.CRC)


def build_selection(
    archive: ZipFile, tasks: Sequence[str], per_task: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    inventory = inventory_selected_tasks(archive, tasks)
    selected: list[dict[str, Any]] = []
    available_counts: dict[str, int] = {}
    for task in tasks:
        available_counts[task] = len(inventory[task])
        for trajectory_id in uniform_sample_ids(tuple(inventory[task]), per_task):
            names = inventory[task][trajectory_id]
            validate_native_member_set(names)
            infos = [archive.getinfo(name) for name in names]
            selected.append(
                {
                    "task": task,
                    "trajectory_id": trajectory_id,
                    "members": [
                        {
                            "path": info.filename,
                            "crc32": f"{int(info.CRC):08x}",
                            "compressed_bytes": int(info.compress_size),
                            "uncompressed_bytes": int(info.file_size),
                            "header_offset": int(info.header_offset),
                        }
                        for info in sorted(infos, key=lambda item: item.filename)
                    ],
                }
            )
    return selected, available_counts


def extract_trajectory(
    *,
    archive: ZipFile,
    resolved_url: str,
    output_root: Path,
    row: dict[str, Any],
    timeout: float,
    retries: int,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    infos = [archive.getinfo(member["path"]) for member in row["members"]]
    missing = [
        info
        for info in infos
        if not _existing_valid(_local_destination(output_root, info.filename), info)
    ]
    downloaded_bytes = 0
    if missing:
        range_start = min(int(info.header_offset) for info in infos)
        range_end = max(_member_end(info) for info in infos)
        response: requests.Response | None = None
        request_session = session or requests.Session()
        for attempt in range(1, retries + 1):
            try:
                response = request_session.get(
                    resolved_url,
                    headers={"Range": f"bytes={range_start}-{range_end}"},
                    timeout=(30, timeout),
                )
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(f"server ignored Range request: {response.status_code}")
                break
            except (requests.RequestException, RuntimeError):
                if attempt == retries:
                    raise
                time.sleep(min(2 ** (attempt - 1), 30))
        assert response is not None
        payload = response.content
        downloaded_bytes = len(payload)
        staged: list[tuple[Path, bytes]] = []
        for info in missing:
            staged.append(
                (
                    _local_destination(output_root, info.filename),
                    _member_bytes(payload, range_start, info),
                )
            )
        for destination, data in staged:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(data)
            temporary.replace(destination)
    receipt_members = []
    for info in sorted(infos, key=lambda item: item.filename):
        destination = _local_destination(output_root, info.filename)
        if not _existing_valid(destination, info):
            raise RuntimeError(f"post-extraction CRC check failed: {info.filename}")
        receipt_members.append(
            {
                "path": info.filename,
                "local_relative_path": destination.relative_to(output_root).as_posix(),
                "bytes": destination.stat().st_size,
                "crc32": f"{int(info.CRC):08x}",
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        )
    return {
        "task": row["task"],
        "trajectory_id": row["trajectory_id"],
        "downloaded_bytes": downloaded_bytes,
        "members": receipt_members,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-index", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--archive-size", type=int, required=True)
    parser.add_argument("--archive-lfs-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--per-task", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.per_task < 1 or args.workers < 1 or args.retries < 1:
        parser.error("per-task, workers, and retries must be positive")

    with ZipFile(args.zip_index.resolve()) as archive:
        selected, available_counts = build_selection(archive, tuple(args.tasks), args.per_task)
        manifest = {
            "schema_version": 1,
            "provenance_class": "official_demonstration_reorganization_and_label_enrichment",
            "source_dataset": "VIMA/VIMA-Data",
            "source_revision": args.dataset_revision,
            "source_url": args.url,
            "archive_size": args.archive_size,
            "archive_lfs_sha256": args.archive_lfs_sha256,
            "selection_method": "uniform_midpoint_quantiles_over_sorted_native_trajectory_ids",
            "per_task": args.per_task,
            "task_order": list(args.tasks),
            "available_counts": available_counts,
            "selected_trajectories": len(selected),
            "selected_compressed_bytes": sum(
                member["compressed_bytes"] for row in selected for member in row["members"]
            ),
            "selected_uncompressed_bytes": sum(
                member["uncompressed_bytes"] for row in selected for member in row["members"]
            ),
            "selection": selected,
            "candidate_g_policy": "absent_not_used_for_selection_or_admission",
            "heldout_policy": "none_inventory_expansion_only",
        }
        args.selection_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.selection_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({key: value for key, value in manifest.items() if key != "selection"}, indent=2), flush=True)
        if args.dry_run:
            return

        resolved = requests.head(args.url, allow_redirects=True, timeout=60).url
        args.output_root.mkdir(parents=True, exist_ok=True)
        thread_state = threading.local()

        def run(row: dict[str, Any]) -> dict[str, Any]:
            if not hasattr(thread_state, "session"):
                thread_state.session = requests.Session()
            return extract_trajectory(
                archive=archive,
                resolved_url=resolved,
                output_root=args.output_root,
                row=row,
                timeout=args.timeout,
                retries=args.retries,
                session=thread_state.session,
            )

        receipts = []
        downloaded = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for index, receipt in enumerate(executor.map(run, selected), start=1):
                receipts.append(receipt)
                downloaded += int(receipt["downloaded_bytes"])
                if index % 100 == 0 or index == len(selected):
                    print(
                        f"[extract] {index}/{len(selected)} transferred_mib={downloaded / 2**20:.1f}",
                        flush=True,
                    )
    receipt_doc = {
        "schema_version": 1,
        "status": "COMPLETE",
        "source_revision": args.dataset_revision,
        "selected_trajectories": len(receipts),
        "transferred_bytes_this_run": downloaded,
        "task_counts": {
            task: sum(row["task"] == task for row in receipts) for task in args.tasks
        },
        "trajectories": receipts,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt_doc, indent=2) + "\n")
    print(json.dumps({key: value for key, value in receipt_doc.items() if key != "trajectories"}, indent=2))


if __name__ == "__main__":
    main()
