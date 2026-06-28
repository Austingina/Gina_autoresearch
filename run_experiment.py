#!/usr/bin/env python3
"""Single experiment entry — eval policy with current experiment.py config."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from experiment import ExperimentConfig, build_pipeline

ROOT = Path(__file__).resolve().parent


def main() -> int:
    p = argparse.ArgumentParser(description="Run one visual guidance field experiment")
    p.add_argument("--task", default="lift")
    p.add_argument("--split", default="s0_fixed")
    p.add_argument("--backend", default="dry_run", choices=["dry_run", "isaac"])
    args = p.parse_args()

    _, _, cfg = build_pipeline(ExperimentConfig())

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cfg.to_dict(), f)
        cfg_path = f.name

    cmd = [
        sys.executable,
        str(ROOT / "prepare.py"),
        "--task",
        args.task,
        "--split",
        args.split,
        "--backend",
        args.backend,
        "--experiment-config",
        cfg_path,
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
