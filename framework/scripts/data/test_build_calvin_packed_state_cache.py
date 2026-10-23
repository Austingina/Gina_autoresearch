#!/usr/bin/env python3
"""Contract test for lossless contiguous CALVIN state packing."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile

import numpy as np


MODULE_PATH = Path(__file__).with_name("build_calvin_packed_state_cache.py")


def main() -> None:
    spec = importlib.util.spec_from_file_location("packed_cache", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        split = Path(tmp) / "training"
        split.mkdir()
        expected = {}
        for index in (3, 7):
            fields = {
                "robot_obs": np.asarray([index, index + 1], dtype=np.float32),
                "scene_obs": np.asarray([index + 2], dtype=np.float32),
                "rel_actions": np.asarray([index + 3, index + 4], dtype=np.float32),
            }
            expected[index] = fields
            np.savez(split / f"episode_{index:07d}.npz", **fields)

        output = split / "state_only_arrays.npz"
        receipt = module.build_cache(split, output)
        assert receipt["frames"] == 2
        assert receipt["frame_min"] == 3
        assert receipt["frame_max"] == 7
        assert len(receipt["cache_sha256"]) == 64
        with np.load(output, allow_pickle=False) as packed:
            assert packed.files == ["frame_index", "robot_obs", "scene_obs", "rel_actions"]
            assert packed["frame_index"].tolist() == [3, 7]
            for row, index in enumerate((3, 7)):
                for field, value in expected[index].items():
                    assert np.array_equal(packed[field][row], value)

        try:
            module.build_cache(split, output)
        except RuntimeError as exc:
            assert "already exists" in str(exc)
        else:
            raise AssertionError("cache overwrite must fail closed")

    print("packed state cache: OK")


if __name__ == "__main__":
    main()
