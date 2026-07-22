#!/usr/bin/env python3
"""Summarize canonical CALVIN experiment runs."""

from __future__ import annotations

import argparse
from pathlib import Path


def read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text(errors="replace").strip()
    except FileNotFoundError:
        return default


def parse_env(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in read_text(path).splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip()
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/nvme33n1/wuk_code/AutoResearch")
    args = ap.parse_args()
    root = Path(args.root)
    runs = sorted((root / "experiments" / "phase1_calvin").glob("*"))
    print("run,state,status,preset,gpu,stdout")
    for run in runs:
        if not run.is_dir():
            continue
        env = parse_env(run / "env.txt")
        state = read_text(run / "state.txt", "unknown")
        status = read_text(run / "status.txt", "")
        stdout = run / "logs" / "stdout.log"
        print(
            f"{run.name},{state},{status},{env.get('preset','')},{env.get('gpu','')},{stdout}"
        )


if __name__ == "__main__":
    main()
