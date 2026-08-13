#!/usr/bin/env python3
"""Regression checks for leakage-free block surface-G calibration."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("calibrate_calvin_block_surface_g.py")
SPEC = importlib.util.spec_from_file_location("calibrate_block_surface_g", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> None:
    assert not set(MODULE.DEFAULT_SOURCE_TASKS) & set(MODULE.DEFAULT_HELDOUT_TASKS)
    assert set(MODULE.DEFAULT_SOURCE_TASKS) | set(MODULE.DEFAULT_HELDOUT_TASKS) == set(
        MODULE.TASKS
    )

    rows = [
        {"task": "push_red_block_left", "error": 1.0},
        {"task": "push_red_block_left", "error": 3.0},
        {"task": "push_blue_block_left", "error": 99.0},
    ]
    summaries = MODULE.grouped_summary(rows, "error")
    assert summaries["push_red_block_left"]["count"] == 2
    assert np.isclose(summaries["push_red_block_left"]["p50"], 2.0)
    assert summaries["push_blue_block_left"]["count"] == 1
    assert summaries["push_red_block_right"]["count"] == 0
    print("block surface-G calibration protocol: OK")


if __name__ == "__main__":
    main()
