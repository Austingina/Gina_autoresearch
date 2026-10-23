#!/usr/bin/env python3
"""Parallel, resumable fixed-range download for one immutable file."""
from __future__ import annotations

import argparse
import importlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-bytes", type=int, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries-per-chunk", type=int, default=30)
    args = parser.parse_args()
    requests = importlib.import_module("requests")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = args.output.stat().st_size if args.output.exists() else 0
    if existing > args.total_bytes:
        raise RuntimeError(f"oversized output: {existing} > {args.total_bytes}")

    def fetch(item: tuple[int, int, int]) -> tuple[int, bytes, str, int]:
        index, start, end = item
        wanted = f"bytes {start}-{end}/{args.total_bytes}"
        expected = end - start + 1
        last_error = None
        for attempt in range(args.retries_per_chunk):
            url = args.url[(index + attempt) % len(args.url)]
            try:
                response = requests.get(
                    url,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=(60, 300),
                )
                declared = response.headers.get("Content-Range", "")
                if response.status_code != 206 or declared != wanted:
                    raise RuntimeError(
                        f"bad range status={response.status_code}, "
                        f"content-range={declared!r}, wanted={wanted!r}"
                    )
                if len(response.content) != expected:
                    raise RuntimeError(
                        f"short range body {len(response.content)} != {expected}"
                    )
                return index, response.content, url, attempt + 1
            except Exception as exc:
                last_error = exc
                time.sleep(min(30, 2 + attempt * 2))
        raise RuntimeError(f"failed range {start}-{end}: {last_error}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while existing < args.total_bytes:
            items = []
            cursor = existing
            for index in range(args.workers):
                if cursor >= args.total_bytes:
                    break
                end = min(cursor + args.chunk_bytes, args.total_bytes) - 1
                items.append((index, cursor, end))
                cursor = end + 1
            results = sorted(pool.map(fetch, items), key=lambda item: item[0])
            with args.output.open("ab") as handle:
                for _, payload, _, _ in results:
                    handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            existing += sum(len(payload) for _, payload, _, _ in results)
            sources = sorted({url for _, _, url, _ in results})
            attempts = max(attempt for _, _, _, attempt in results)
            print(
                f"BATCH_OK bytes={existing}/{args.total_bytes} "
                f"chunks={len(results)} max_attempt={attempts} sources={sources}",
                flush=True,
            )

    print(f"DOWNLOAD_COMPLETE bytes={existing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
