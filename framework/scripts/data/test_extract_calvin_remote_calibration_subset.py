#!/usr/bin/env python3
"""Regression checks for deterministic, balanced CALVIN range selection."""

from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np


if importlib.util.find_spec("remotezip") is None:
    remotezip_stub = types.ModuleType("remotezip")
    remotezip_stub.RemoteZip = object
    sys.modules["remotezip"] = remotezip_stub
if importlib.util.find_spec("requests") is None:
    sys.modules["requests"] = types.ModuleType("requests")

MODULE_PATH = Path(__file__).with_name("extract_calvin_remote_calibration_subset.py")
SPEC = importlib.util.spec_from_file_location("extract_calvin_subset", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> None:
    scene_info = {
        "calvin_scene_A": [0, 999],
        "calvin_scene_B": [1000, 1999],
        "calvin_scene_C": [2000, 2999],
    }
    tasks: list[str] = []
    intervals: list[list[int]] = []
    for task_index, task in enumerate(MODULE.SOURCE_TASKS + MODULE.HELDOUT_TASKS):
        for scene_index in range(3):
            base = scene_index * 1000 + task_index * 100
            starts = [base + 5, base + 21, base + 42]
            for start in starts:
                tasks.append(task)
                intervals.append([start, start + 9])
    annotation = {
        "language": {
            "task": tasks,
            "ann": [f"annotation-{index}" for index in range(len(tasks))],
            "emb": np.arange(len(tasks) * 4, dtype=np.float32).reshape(-1, 1, 4),
        },
        "info": {"indx": intervals},
    }
    selected = MODULE.select_segments(
        annotation,
        scene_info,
        source_per_scene=2,
        heldout_per_scene=1,
        validation_modulus=5,
        seed=7,
    )
    assert len(selected) == 30
    assert len({(row["start"], row["end"]) for row in selected}) == len(selected)
    assert all(
        tasks[int(row["annotation_index"])] == row["task"] for row in selected
    )
    assert Counter(row["task"] for row in selected) == {
        **{task: 6 for task in MODULE.SOURCE_TASKS},
        **{task: 3 for task in MODULE.HELDOUT_TASKS},
    }
    assert Counter(row["scene"] for row in selected) == {
        "calvin_scene_A": 10,
        "calvin_scene_B": 10,
        "calvin_scene_C": 10,
    }
    infos = [
        types.SimpleNamespace(header_offset=offset, compress_size=600, filename=name, extra=b"")
        for offset, name in ((0, "a"), (800, "b"), (1600, "c"))
    ]
    bounded_groups = MODULE.physical_groups(infos, max_range_bytes=1500)
    assert [len(group) for group in bounded_groups] == [2, 1]
    print("CALVIN range subset selection: OK")


if __name__ == "__main__":
    main()
