#!/usr/bin/env python3
"""Build a sparse local ZIP index from a remotely fetched central-directory tail."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tail", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-size", type=int, required=True)
    parser.add_argument("--central-directory-offset", type=int, required=True)
    parser.add_argument("--expected-entries", type=int)
    args = parser.parse_args()

    tail = args.tail.read_bytes()
    expected_tail_size = args.archive_size - args.central_directory_offset
    if len(tail) != expected_tail_size:
        raise RuntimeError(
            f"tail size mismatch: got {len(tail)}, expected {expected_tail_size}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        handle.truncate(args.archive_size)
        handle.seek(args.central_directory_offset)
        handle.write(tail)

    with ZipFile(args.output) as archive:
        entries = len(archive.infolist())
    if args.expected_entries is not None and entries != args.expected_entries:
        raise RuntimeError(
            f"entry count mismatch: got {entries}, expected {args.expected_entries}"
        )
    print(
        f"[index] path={args.output} logical_bytes={args.archive_size} "
        f"physical_tail_bytes={len(tail)} entries={entries}"
    )


if __name__ == "__main__":
    main()
