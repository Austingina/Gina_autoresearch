#!/usr/bin/env python3
"""Regression test for NumPy-2-authored CALVIN annotation pickles on NumPy 1.x."""

from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, SCRIPT_DIR.as_posix())

import train_eval_state_bc_task_policy as base  # noqa: E402


def main() -> None:
    split_dir = (
        REPO_ROOT
        / "data/calvin/task_ABC_D_g_intent_factorial_s12_h3/training"
    )
    segments = base.load_lang_segments(split_dir)
    assert len(segments) == 324
    assert {str(segment["task"]) for segment in segments}
    print("NumPy pickle compatibility: OK")


if __name__ == "__main__":
    main()
