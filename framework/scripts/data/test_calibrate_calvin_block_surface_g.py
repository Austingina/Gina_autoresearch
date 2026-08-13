#!/usr/bin/env python3
"""Regression checks for leakage-free block surface-G calibration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import types

import numpy as np


MODULE_PATH = Path(__file__).with_name("calibrate_calvin_block_surface_g.py")
if importlib.util.find_spec("yaml") is None:
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda value: __import__("json").loads(value)
    sys.modules["yaml"] = yaml_stub
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

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        scene_dir = root / "conf" / "scene"
        scene_dir.mkdir(parents=True)
        scene_yaml = """{
          "global_scaling": 1.0,
          "objects": {
            "fixed_objects": {},
            "movable_objects": {
              "block_red": {"file": "blocks/red.urdf"},
              "block_blue": {"file": "blocks/blue.urdf"},
              "block_pink": {"file": "blocks/pink.urdf"}
            }
          }
        }"""
        for name in ("calvin_scene_A", "calvin_scene_B"):
            (scene_dir / f"{name}.yaml").write_text(scene_yaml)
        for name in ("red", "blue", "pink"):
            urdf = root / "data" / "blocks" / f"{name}.urdf"
            urdf.parent.mkdir(parents=True, exist_ok=True)
            urdf.write_text(
                '<robot><link><collision><geometry><box size="0.1 0.1 0.1"/>'
                "</geometry></collision></link></robot>"
            )
        split = root / "split"
        split.mkdir()
        np.save(
            split / "scene_info.npy",
            {"calvin_scene_A": [0, 9], "calvin_scene_B": [10, 19]},
            allow_pickle=True,
        )
        geometries = MODULE.load_scene_geometries(split, root)
        assert MODULE.geometry_for_segment(geometries, 2, 7)["name"] == "calvin_scene_A"
        assert MODULE.geometry_for_segment(geometries, 12, 17)["name"] == "calvin_scene_B"
        try:
            MODULE.geometry_for_segment(geometries, 8, 11)
        except RuntimeError:
            pass
        else:
            raise AssertionError("A segment spanning scene ranges must be rejected")
    print("block surface-G calibration protocol: OK")


if __name__ == "__main__":
    main()
