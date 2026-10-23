#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import tempfile
import numpy as np

SCRIPT = Path(__file__).with_name("build_calvin_fixed_scene_y_push_resets.py")


def load_module():
    spec = importlib.util.spec_from_file_location("build_calvin_fixed_scene_y_push_resets", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def candidate(color: str, y: float, index: int):
    scene = np.zeros(24, dtype=np.float32)
    order = [color] + [name for name in ("red", "blue", "pink") if name != color]
    xs = {order[0]: 0.06, order[1]: -0.10, order[2]: 0.139}
    ys = {order[0]: y, order[1]: -0.13, order[2]: 0.10}
    positions = {name: (xs[name], ys[name], 0.46) for name in ("red", "blue", "pink")}
    for name, sl in {"red": slice(6, 9), "blue": slice(12, 15), "pink": slice(18, 21)}.items():
        scene[sl] = positions[name]
    return {"target_color": color, "source_sample_index": index, "robot_obs": np.zeros(15), "scene_obs": scene}


def test_target_specific_y_margin_and_task_pair(module):
    rows = [candidate(color, -0.04, i) for i, color in enumerate(module.COLORS)]
    manifest = module.build_manifest(rows, seed=7, count_per_color=1)
    assert manifest["count"] == 3
    for row in manifest["resets"]:
        color = row["target_color"]
        assert row["eligible_tasks"] == [f"push_{color}_block_forward", f"push_{color}_block_backward"]
        y = row["object_positions"][f"block_{color}"][1]
        assert module.TARGET_Y_BOUNDS[0] <= y <= module.TARGET_Y_BOUNDS[1]


def test_rejects_target_without_bidirectional_margin(module):
    row = candidate("red", 0.08, 0)
    try:
        module.validate_candidate(row["scene_obs"], target_color="red")
    except ValueError:
        return
    raise AssertionError("target outside central Y margin was accepted")


def test_manifest_roundtrip_is_json_serializable(module):
    rows = [candidate(color, -0.04, i) for i, color in enumerate(module.COLORS)]
    payload = module.build_manifest(rows, seed=7, count_per_color=1)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "resets.json"
        module.write_json_atomic(path, payload)
        loaded = json.loads(path.read_text())
    assert loaded["factor_axes"]["direction"] == ["forward", "backward"]


def test_reset_id_namespace_prevents_cross_pool_collisions(module):
    rows = [candidate(color, -0.04, i) for i, color in enumerate(module.COLORS)]
    first = module.build_manifest(rows, seed=7, count_per_color=1, id_namespace="s7")
    second = module.build_manifest(rows, seed=8, count_per_color=1, id_namespace="s8")
    first_ids = {row["reset_id"] for row in first["resets"]}
    second_ids = {row["reset_id"] for row in second["resets"]}
    assert first_ids.isdisjoint(second_ids)
    assert first["id_namespace"] == "s7"


def test_skip_per_color_selects_new_physical_candidates(module):
    rows = []
    for color in module.COLORS:
        rows.extend([candidate(color, -0.04, 10), candidate(color, -0.04, 20)])
    manifest = module.build_manifest(rows, seed=7, count_per_color=1, skip_per_color=1, id_namespace="continued")
    assert {row["source_sample_index"] for row in manifest["resets"]} == {20}
    assert manifest["skip_per_color"] == 1


def main():
    module = load_module()
    test_target_specific_y_margin_and_task_pair(module)
    test_rejects_target_without_bidirectional_margin(module)
    test_manifest_roundtrip_is_json_serializable(module)
    test_reset_id_namespace_prevents_cross_pool_collisions(module)
    test_skip_per_color_selects_new_physical_candidates(module)
    print("CALVIN fixed-scene Y-push reset pool unit contract: OK")


if __name__ == "__main__":
    main()
