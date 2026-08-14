#!/usr/bin/env python3
"""Regression checks for matched interaction-volume G relabeling."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import types

import numpy as np


MODULE_PATH = Path(__file__).with_name("build_calvin_interaction_volume_cache.py")
if importlib.util.find_spec("yaml") is None:
    sys.modules["yaml"] = types.ModuleType("yaml")
SPEC = importlib.util.spec_from_file_location("build_volume_cache", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        split = Path(tmp_dir)
        for index in range(4):
            robot = np.zeros(15, dtype=np.float32)
            robot[:3] = np.asarray([0.2 - index * 0.02, 0.0, 0.0])
            scene = np.zeros(24, dtype=np.float32)
            scene[6:9] = np.asarray([0.1 + max(0, index - 1) * 0.0021, 0.0, 0.0])
            np.savez(
                split / f"episode_{index:07d}.npz",
                robot_obs=robot,
                scene_obs=scene,
                rel_actions=np.full(7, index, dtype=np.float32),
            )
        geometry = {
            "name": "calvin_scene_test",
            "slices": {"red": slice(6, 9)},
            "half_extents": {"red": np.full(3, 0.02, dtype=np.float32)},
        }
        arrays, metadata = MODULE.relabel_segment(
            split,
            "push_red_block_right",
            0,
            3,
            geometry,
            tool_clearance=0.018,
            motion_threshold=0.002,
            attraction_mode="sticky_post_motion_zero",
        )
        assert metadata["motion_onset_local_index"] == 2
        assert np.allclose(arrays["minimal_g"][0, :3], [-0.062, 0.0, 0.0])
        assert np.linalg.norm(arrays["minimal_g"][1, :3]) > 0
        assert np.allclose(arrays["minimal_g"][2:, :3], 0)
        assert np.allclose(arrays["minimal_g"][:, 3:6], [1.0, 0.0, 0.0])
        closed_loop, closed_loop_metadata = MODULE.relabel_segment(
            split,
            "push_red_block_right",
            0,
            3,
            geometry,
            tool_clearance=0.018,
            motion_threshold=0.002,
            attraction_mode="closed_loop_volume",
        )
        assert closed_loop_metadata["attraction_mode"] == "closed_loop_volume"
        assert np.linalg.norm(closed_loop["minimal_g"][2, :3]) > 0
        assert np.allclose(closed_loop["minimal_g"][3, :3], 0)
    print("matched interaction-volume G cache: OK")


if __name__ == "__main__":
    main()
