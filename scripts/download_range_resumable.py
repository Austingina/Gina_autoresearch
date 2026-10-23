#!/usr/bin/env python3
"""Resume a large immutable file with verified fixed-size HTTP ranges."""
from __future__ import annotations

import argparse
import importlib
import os
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-bytes", type=int, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--retries-per-chunk", type=int, default=30)
    args = parser.parse_args()

    requests = importlib.import_module("requests")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = args.output.stat().st_size if args.output.exists() else 0
    if existing > args.total_bytes:
        raise RuntimeError(f"output is oversized: {existing} > {args.total_bytes}")

    session = requests.Session()
    while existing < args.total_bytes:
        end = min(existing + args.chunk_bytes, args.total_bytes) - 1
        expected = end - existing + 1
        completed = False
        last_error = None
        for attempt in range(args.retries_per_chunk):
            url = args.url[attempt % len(args.url)]
            try:
                response = session.get(
                    url,
                    headers={"Range": f"bytes={existing}-{end}"},
                    timeout=(60, 300),
                )
                declared = response.headers.get("Content-Range", "")
                wanted = f"bytes {existing}-{end}/{args.total_bytes}"
                if response.status_code != 206 or declared != wanted:
                    raise RuntimeError(
                        f"bad range response status={response.status_code} "
                        f"content-range={declared!r}, wanted={wanted!r}"
                    )
                payload = response.content
                if len(payload) != expected:
                    raise RuntimeError(
                        f"short range body: {len(payload)} != {expected}"
                    )
                with args.output.open("ab") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                existing += len(payload)
                print(
                    f"RANGE_OK bytes={existing}/{args.total_bytes} "
                    f"attempt={attempt + 1} source={url}",
                    flush=True,
                )
                completed = True
                break
            except Exception as exc:  # retry the exact immutable byte range
                last_error = exc
                print(
                    f"RANGE_RETRY start={existing} end={end} "
                    f"attempt={attempt + 1} error={exc}",
                    flush=True,
                )
                time.sleep(min(30, 2 + attempt * 2))
        if not completed:
            raise RuntimeError(
                f"failed immutable range {existing}-{end}: {last_error}"
            )

    print(f"DOWNLOAD_COMPLETE bytes={existing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
