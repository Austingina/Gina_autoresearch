#!/usr/bin/env python3
"""TDD contract for native fixed-scene CALVIN push reset pools."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO / "framework/scripts/data/build_calvin_fixed_scene_push_resets.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_calvin_fixed_scene_push_resets", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scene_obs(red, blue, pink):
    value = np.zeros(24, dtype=np.float32)
    value[6:9] = red; value[12:15] = blue; value[18:21] = pink
    return value


def test_candidate_requires_all_scene_d_objects_in_shared_workspace(module):
    valid = scene_obs([-0.06, -0.04, 0.46], [0.00, -0.14, 0.46], [0.06, 0.02, 0.46])
    assert module.validate_candidate(valid) == {
        "block_red": [-0.06, -0.04, 0.46],
        "block_blue": [0.0, -0.14, 0.46],
        "block_pink": [0.06, 0.02, 0.46],
    }
    outside = valid.copy(); outside[12] = 0.20
    try:
        module.validate_candidate(outside)
    except ValueError as exc:
        assert "workspace" in str(exc)
    else:
        raise AssertionError("out-of-workspace reset must be rejected")


def test_candidate_rejects_colliding_or_wrong_height_objects(module):
    colliding = scene_obs([0.0, -0.1, 0.46], [0.01, -0.1, 0.46], [0.07, 0.02, 0.46])
    for invalid, word in [
        (colliding, "separation"),
        (scene_obs([-0.06, -0.04, 0.46], [0.0, -0.14, 0.52], [0.06, 0.02, 0.46]), "height"),
    ]:
        try:
            module.validate_candidate(invalid)
        except ValueError as exc:
            assert word in str(exc)
        else:
            raise AssertionError("invalid reset must be rejected")


def test_manifest_is_fixed_scene_balanced_and_reusable_across_tasks(module):
    robot = np.zeros(15, dtype=np.float32)
    candidates = []
    for index in range(3):
        obs = scene_obs([-0.06, -0.04 + index * 0.002, 0.46], [0.0, -0.14, 0.46], [0.06, 0.02, 0.46])
        candidates.append({"robot_obs": robot + index, "scene_obs": obs})
    manifest = module.build_manifest(candidates, seed=11, count=2)
    assert manifest["format"] == "calvin_fixed_scene_push_resets_v1"
    assert manifest["runtime_scene"] == "calvin_scene_D"
    assert len(manifest["resets"]) == 2
    assert manifest["tasks"] == list(module.PUSH_TASKS)
    for reset in manifest["resets"]:
        assert reset["scene"] == "calvin_scene_D"
        assert len(reset["robot_obs"]) == 15
        assert len(reset["scene_obs"]) == 24
        assert set(reset["eligible_tasks"]) == set(module.PUSH_TASKS)


def main():
    module = load_module()
    test_candidate_requires_all_scene_d_objects_in_shared_workspace(module)
    test_candidate_rejects_colliding_or_wrong_height_objects(module)
    test_manifest_is_fixed_scene_balanced_and_reusable_across_tasks(module)
    print("CALVIN fixed-scene reset pool unit contract: OK")


if __name__ == "__main__":
    main()
