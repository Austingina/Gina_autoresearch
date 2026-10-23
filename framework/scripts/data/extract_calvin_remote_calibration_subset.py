#!/usr/bin/env python3
"""Range-extract a balanced calibration subset from the official CALVIN ZIP."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import struct
import time
from typing import Any, Sequence
import zlib
from zipfile import ZipFile

import numpy as np
import requests
from remotezip import RemoteZip


SOURCE_TASKS = (
    "push_red_block_left",
    "push_blue_block_right",
    "push_pink_block_left",
    "push_pink_block_right",
)
HELDOUT_TASKS = ("push_blue_block_left", "push_red_block_right")
LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
LOCAL_FILE_SIGNATURE = 0x04034B50


def scene_for_interval(scene_info: dict[str, list[int]], start: int, end: int) -> str:
    matches = [
        name
        for name, bounds in scene_info.items()
        if int(bounds[0]) <= start <= end <= int(bounds[1])
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Interval [{start}, {end}] crosses scene ranges")
    return matches[0]


def overlaps(interval: tuple[int, int], selected: list[tuple[int, int]]) -> bool:
    start, end = interval
    return any(not (end < other_start or start > other_end) for other_start, other_end in selected)


def choose_nonoverlapping(
    candidates: list[dict[str, object]],
    count: int,
    rng: random.Random,
    selected_intervals: list[tuple[int, int]],
) -> list[dict[str, object]]:
    candidates = list(candidates)
    rng.shuffle(candidates)
    chosen = []
    for row in candidates:
        interval = (int(row["start"]), int(row["end"]))
        if overlaps(interval, selected_intervals):
            continue
        chosen.append(row)
        selected_intervals.append(interval)
        if len(chosen) == count:
            return chosen
    raise RuntimeError(f"Only selected {len(chosen)}/{count} non-overlapping intervals")


def select_segments(
    annotation: dict[str, object],
    scene_info: dict[str, list[int]],
    source_per_scene: int,
    heldout_per_scene: int,
    validation_modulus: int,
    seed: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for annotation_index, (task, interval) in enumerate(
        zip(annotation["language"]["task"], annotation["info"]["indx"])
    ):
        task = str(task)
        if task not in SOURCE_TASKS + HELDOUT_TASKS:
            continue
        start, end = (int(interval[0]), int(interval[1]))
        scene = scene_for_interval(scene_info, start, end)
        grouped[(task, scene)].append(
            {
                "task": task,
                "scene": scene,
                "start": start,
                "end": end,
                "annotation_index": annotation_index,
            }
        )

    rng = random.Random(seed)
    selected: list[dict[str, object]] = []
    selected_intervals: list[tuple[int, int]] = []
    scenes = sorted(scene_info)
    for task in SOURCE_TASKS:
        for scene in scenes:
            candidates = grouped[(task, scene)]
            validation = [
                row for row in candidates if int(row["start"]) % validation_modulus == 0
            ]
            train = [
                row for row in candidates if int(row["start"]) % validation_modulus != 0
            ]
            selected.extend(
                choose_nonoverlapping(validation, 1, rng, selected_intervals)
            )
            selected.extend(
                choose_nonoverlapping(
                    train, source_per_scene - 1, rng, selected_intervals
                )
            )
    for task in HELDOUT_TASKS:
        for scene in scenes:
            selected.extend(
                choose_nonoverlapping(
                    grouped[(task, scene)], heldout_per_scene, rng, selected_intervals
                )
            )
    return sorted(selected, key=lambda row: int(row["start"]))


def member_bytes_from_range(payload: bytes, range_start: int, info: Any) -> bytes:
    offset = int(info.header_offset) - range_start
    header = LOCAL_HEADER.unpack_from(payload, offset)
    signature, _version, _flags, method, _mtime, _mdate, _crc, _csize, _usize, name_len, extra_len = header
    if signature != LOCAL_FILE_SIGNATURE:
        raise RuntimeError(f"Invalid local header for {info.filename}")
    data_start = offset + LOCAL_HEADER.size + name_len + extra_len
    compressed = payload[data_start : data_start + int(info.compress_size)]
    if len(compressed) != int(info.compress_size):
        raise RuntimeError(f"Truncated compressed data for {info.filename}")
    if method == 0:
        data = compressed
    elif method == 8:
        data = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise RuntimeError(f"Unsupported ZIP method {method} for {info.filename}")
    if len(data) != int(info.file_size):
        raise RuntimeError(f"Uncompressed size mismatch for {info.filename}")
    if zlib.crc32(data) & 0xFFFFFFFF != int(info.CRC):
        raise RuntimeError(f"CRC mismatch for {info.filename}")
    return data


def extract_member_from_range(
    payload: bytes, range_start: int, info: Any, destination: Path
) -> None:
    data = member_bytes_from_range(payload, range_start, info)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


def fetch_member(session: requests.Session, url: str, info: Any, timeout: float) -> bytes:
    range_start = int(info.header_offset)
    range_end = member_end(info)
    response = session.get(
        url,
        headers={"Range": f"bytes={range_start}-{range_end}"},
        timeout=timeout,
    )
    response.raise_for_status()
    if response.status_code != 206:
        raise RuntimeError(f"Server ignored Range request: {response.status_code}")
    return member_bytes_from_range(response.content, range_start, info)


def member_end(info: Any) -> int:
    return (
        int(info.header_offset)
        + LOCAL_HEADER.size
        + len(info.filename.encode())
        + len(info.extra)
        + int(info.compress_size)
        + 64
    )


def physical_groups(
    infos: Sequence[Any],
    max_gap: int = 1 << 20,
    max_range_bytes: int | None = None,
) -> list[list[Any]]:
    ordered = sorted(infos, key=lambda info: int(info.header_offset))
    groups: list[list[Any]] = []
    for info in ordered:
        gap_too_large = bool(
            groups and int(info.header_offset) - member_end(groups[-1][-1]) > max_gap
        )
        range_too_large = bool(
            groups
            and max_range_bytes is not None
            and member_end(info) - int(groups[-1][0].header_offset) + 1 > max_range_bytes
        )
        if not groups or gap_too_large or range_too_large:
            groups.append([info])
        else:
            groups[-1].append(info)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-per-task-scene", type=int, default=8)
    parser.add_argument("--heldout-per-task-scene", type=int, default=3)
    parser.add_argument("--validation-modulus", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--zip-index",
        type=Path,
        help="Sparse local ZIP containing the central directory; avoids remote index scanning",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help="Optional JSON path for the full deterministic selection, including segment detail",
    )
    args = parser.parse_args()
    if args.source_per_task_scene < 2:
        parser.error("source-per-task-scene must be at least 2")
    if args.workers < 1:
        parser.error("workers must be at least 1")
    if args.retries < 1:
        parser.error("retries must be at least 1")

    root = args.output_dir.resolve()
    split = root / "training"
    archive_prefix = "task_ABC_D/"

    def destination_for(filename: str) -> Path:
        if not filename.startswith(archive_prefix):
            raise RuntimeError(f"Archive member is outside {archive_prefix}: {filename}")
        return root / filename[len(archive_prefix) :]

    metadata_names = (
        "task_ABC_D/training/lang_annotations/auto_lang_ann.npy",
        "task_ABC_D/training/.hydra/merged_config.yaml",
        "task_ABC_D/training/scene_info.npy",
    )
    archive_context = ZipFile(args.zip_index.resolve()) if args.zip_index else RemoteZip(args.url)
    with archive_context as archive:
        session = requests.Session()
        if args.zip_index:
            metadata = {
                name: fetch_member(session, args.url, archive.getinfo(name), args.timeout)
                for name in metadata_names
            }
        else:
            metadata = {name: archive.read(name) for name in metadata_names}
        annotation = np.load(
            __import__("io").BytesIO(metadata[metadata_names[0]]), allow_pickle=True
        ).item()
        scene_info = np.load(
            __import__("io").BytesIO(metadata[metadata_names[2]]), allow_pickle=True
        ).item()
        segments = select_segments(
            annotation,
            scene_info,
            args.source_per_task_scene,
            args.heldout_per_task_scene,
            args.validation_modulus,
            args.seed,
        )
        selected_frames = sum(int(row["end"]) - int(row["start"]) + 1 for row in segments)
        selected_compressed = sum(
            int(archive.getinfo(f"task_ABC_D/training/episode_{frame:07d}.npz").compress_size)
            for row in segments
            for frame in range(int(row["start"]), int(row["end"]) + 1)
        )
        selection_summary = {
            "segments": len(segments),
            "frames": selected_frames,
            "compressed_bytes": selected_compressed,
            "task_counts": dict(
                sorted(Counter(str(row["task"]) for row in segments).items())
            ),
            "scene_counts": dict(
                sorted(Counter(str(row["scene"]) for row in segments).items())
            ),
        }
        if args.selection_manifest:
            args.selection_manifest.parent.mkdir(parents=True, exist_ok=True)
            args.selection_manifest.write_text(
                json.dumps(
                    {
                        **selection_summary,
                        "source_url": args.url,
                        "selection_seed": args.seed,
                        "source_tasks": list(SOURCE_TASKS),
                        "heldout_tasks": list(HELDOUT_TASKS),
                        "source_per_task_scene": args.source_per_task_scene,
                        "heldout_per_task_scene": args.heldout_per_task_scene,
                        "validation_modulus": args.validation_modulus,
                        "segments_detail": segments,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        if args.dry_run:
            print(json.dumps(selection_summary, indent=2))
            return

        def extract_segment(
            item: tuple[int, dict[str, object]],
        ) -> tuple[int, dict[str, object], int, int, int]:
            index, segment = item
            frame_names = [
                f"task_ABC_D/training/episode_{frame:07d}.npz"
                for frame in range(int(segment["start"]), int(segment["end"]) + 1)
            ]
            infos = [archive.getinfo(name) for name in frame_names]
            missing_infos = [
                info
                for info in infos
                if not (
                    destination_for(info.filename).exists()
                    and destination_for(info.filename).stat().st_size
                    == int(info.file_size)
                )
            ]
            groups = physical_groups(missing_infos)
            segment_downloaded = 0
            worker_session = requests.Session()
            for group in groups:
                range_start = int(group[0].header_offset)
                range_end = member_end(group[-1])
                response: requests.Response | None = None
                for attempt in range(1, args.retries + 1):
                    try:
                        response = worker_session.get(
                            args.url,
                            headers={"Range": f"bytes={range_start}-{range_end}"},
                            timeout=args.timeout,
                        )
                        response.raise_for_status()
                        if response.status_code != 206:
                            raise RuntimeError(
                                f"Server ignored Range request: {response.status_code}"
                            )
                        break
                    except (requests.RequestException, RuntimeError):
                        if attempt == args.retries:
                            raise
                        time.sleep(min(2 ** (attempt - 1), 30))
                assert response is not None
                payload = response.content
                segment_downloaded += len(payload)
                for info in group:
                    destination = destination_for(info.filename)
                    extract_member_from_range(payload, range_start, info, destination)
            return index, segment, len(infos), len(groups), segment_downloaded

        downloaded = 0
        frame_count = 0
        indexed_segments = list(enumerate(segments, start=1))
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            results = executor.map(extract_segment, indexed_segments)
            for index, segment, segment_frames, group_count, segment_downloaded in results:
                downloaded += segment_downloaded
                frame_count += segment_frames
                print(
                    f"[extract] {index}/{len(segments)} {segment['scene']} "
                    f"{segment['task']} frames={segment_frames} ranges={group_count} "
                    f"total_mib={downloaded / 2**20:.1f}",
                    flush=True,
                )

    (split / ".hydra").mkdir(parents=True, exist_ok=True)
    (split / ".hydra" / "merged_config.yaml").write_bytes(metadata[metadata_names[1]])
    np.save(split / "scene_info.npy", scene_info, allow_pickle=True)
    language_dir = split / "lang_annotations"
    language_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        language_dir / "auto_lang_ann.npy",
        {
            "language": {
                "ann": [
                    str(annotation["language"]["ann"][int(row["annotation_index"])])
                    for row in segments
                ],
                "task": [str(row["task"]) for row in segments],
                "emb": np.stack(
                    [
                        np.asarray(
                            annotation["language"]["emb"][int(row["annotation_index"])],
                            dtype=np.float32,
                        )
                        for row in segments
                    ]
                ),
            },
            "info": {
                "indx": np.asarray(
                    [[int(row["start"]), int(row["end"])] for row in segments],
                    dtype=np.int64,
                )
            },
        },
        allow_pickle=True,
    )
    manifest = {
        "source_url": args.url,
        "selection_seed": args.seed,
        "source_tasks": list(SOURCE_TASKS),
        "heldout_tasks": list(HELDOUT_TASKS),
        "source_per_task_scene": args.source_per_task_scene,
        "heldout_per_task_scene": args.heldout_per_task_scene,
        "validation_modulus": args.validation_modulus,
        "segments": len(segments),
        "frames": frame_count,
        "downloaded_bytes": downloaded,
        "task_counts": dict(sorted(Counter(str(row["task"]) for row in segments).items())),
        "scene_counts": dict(sorted(Counter(str(row["scene"]) for row in segments).items())),
        "segments_detail": segments,
    }
    (root / "calibration_subset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "segments_detail"}, indent=2))


if __name__ == "__main__":
    main()
