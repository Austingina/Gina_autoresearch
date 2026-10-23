#!/usr/bin/env python3
"""Extract a task-minimal subset from RoboTwin's official objects.zip.

RemoteZip delegates decompression and CRC verification to Python's ZipFile.
The emitted manifest adds SHA-256 for every extracted regular file.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prefix", action="append", required=True)
    args = parser.parse_args()

    root = args.output_root.resolve()
    rows = []
    remote_zip = importlib.import_module("remotezip").RemoteZip
    with remote_zip(args.url, initial_buffer_size=256 * 1024) as archive:
        infos = [
            info
            for info in archive.infolist()
            if any(info.filename.startswith(prefix.rstrip("/") + "/") for prefix in args.prefix)
        ]
        found_prefixes = {
            prefix: any(info.filename.startswith(prefix.rstrip("/") + "/") for info in infos)
            for prefix in args.prefix
        }
        if not all(found_prefixes.values()):
            raise RuntimeError(f"missing required archive prefixes: {found_prefixes}")

        for info in infos:
            relative = PurePosixPath(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe zip member: {info.filename}")
            destination = root.joinpath(*relative.parts)
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            payload = archive.read(info)  # ZipExtFile validates declared CRC32.
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(payload)
            temporary.replace(destination)
            rows.append(
                {
                    "archive_member": info.filename,
                    "bytes": len(payload),
                    "crc32": f"{info.CRC:08x}",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "destination": str(destination),
                }
            )

    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": args.url,
        "archive_lfs_sha256": "4da608668a06544005effc986bff43540a208b45c34433e064ddf95261420182",
        "prefixes": args.prefix,
        "regular_file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("regular_file_count", "total_bytes", "prefixes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
