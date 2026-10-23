#!/usr/bin/env python3
"""TDD contract for parameterized CALVIN block-push collection."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np

REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO / "framework/scripts/data/collect_calvin_parametric_push_expert.py"


def load_module():
    spec = importlib.util.spec_from_file_location("collect_calvin_parametric_push_expert", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_task_parser_covers_full_factorial(module):
    expected = {
        f"push_{color}_block_{direction}"
        for color in ("red", "blue", "pink")
        for direction in ("left", "right")
    }
    assert set(module.PUSH_TASKS) == expected
    left = module.parse_push_task("push_red_block_left")
    right = module.parse_push_task("push_blue_block_right")
    assert left["object_name"] == "block_red"
    np.testing.assert_allclose(left["flow"], [-1.0, 0.0, 0.0])
    assert right["object_name"] == "block_blue"
    np.testing.assert_allclose(right["flow"], [1.0, 0.0, 0.0])


def test_controller_mirrors_contact_side_and_push_direction(module):
    pos = np.asarray([0.10, -0.15, 0.46], dtype=np.float32)
    left = module.ParametricPushExpert("push_red_block_left", pos)
    right = module.ParametricPushExpert("push_red_block_right", pos)
    assert left.waypoints["contact"][0] > pos[0]
    assert right.waypoints["contact"][0] < pos[0]
    assert left.waypoints["push"][0] < left.waypoints["contact"][0]
    assert right.waypoints["push"][0] > right.waypoints["contact"][0]
    np.testing.assert_allclose(left.waypoints["contact"][1:], right.waypoints["contact"][1:])


def test_parser_and_controller_cover_world_y_directions(module):
    forward = module.parse_push_task("push_red_block_forward")
    backward = module.parse_push_task("push_blue_block_backward")
    np.testing.assert_allclose(forward["flow"], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(backward["flow"], [0.0, -1.0, 0.0])
    pos = np.asarray([0.08, -0.02, 0.46], dtype=np.float32)
    away = module.ParametricPushExpert("push_red_block_forward", pos)
    toward = module.ParametricPushExpert("push_red_block_backward", pos)
    assert away.gripper_action == -1.0
    assert away.waypoints["contact"][1] < pos[1]
    assert toward.waypoints["contact"][1] > pos[1]
    assert away.waypoints["push"][1] > away.waypoints["contact"][1]
    assert toward.waypoints["push"][1] < toward.waypoints["contact"][1]


def test_y_controller_tracks_live_object_with_lateral_error_correction(module):
    initial = np.asarray([0.08, -0.04, 0.46], dtype=np.float32)
    expert = module.ParametricPushExpert("push_red_block_forward", initial)
    assert expert.waypoints["contact"][2] <= initial[2] + 0.006
    expert.state = "push"
    robot = np.zeros(15, dtype=np.float32)
    robot[:3] = [0.11, -0.07, 0.465]
    robot[3:6] = expert.orientation
    object_pos = np.asarray([0.08, -0.01, 0.46], dtype=np.float32)
    command = expert.command(robot, object_pos=object_pos, oracle_success=False)
    assert command["state"] == "push"
    assert command["target_pos"][0] < object_pos[0]
    assert 0.005 < object_pos[1] - command["target_pos"][1] < 0.04
    assert command["target_pos"][2] <= initial[2] + 0.006

    robot[:3] = [0.11, -0.07, 0.465]
    drifted_object = np.asarray([0.12, -0.01, 0.46], dtype=np.float32)
    command = expert.command(robot, object_pos=drifted_object, oracle_success=False)
    assert command["target_pos"][0] < initial[0]
    assert command["target_pos"][0] > 0.065

    robot[:3] = [0.05, -0.07, 0.465]
    command = expert.command(robot, object_pos=object_pos, oracle_success=False)
    assert command["target_pos"][0] > object_pos[0]


def test_y_controller_uses_direction_and_object_specific_settle_parameters(module):
    pos = np.asarray([0.08, -0.04, 0.46], dtype=np.float32)
    red_forward = module.ParametricPushExpert("push_red_block_forward", pos)
    pink_forward = module.ParametricPushExpert("push_pink_block_forward", pos)
    red_backward = module.ParametricPushExpert("push_red_block_backward", pos)
    assert np.isclose(red_forward.y_push_standoff, 0.018)
    assert np.isclose(red_forward.settle_progress, 0.095)
    assert np.isclose(pink_forward.y_push_standoff, 0.005)
    assert np.isclose(pink_forward.settle_progress, 0.090)
    assert np.isclose(red_backward.y_push_standoff, 0.018)
    assert np.isclose(red_backward.settle_progress, 0.102)
    assert not red_forward.should_settle(0.090)
    assert red_forward.should_settle(0.096)
    assert red_forward.should_settle(0.096)
    assert [pink_forward.should_settle(0.092) for _ in range(4)] == [True, True, True, True]
    assert not pink_forward.should_settle(0.092)
    assert pink_forward.should_settle(0.103)
    assert not red_backward.should_settle(0.090)
    assert red_backward.should_settle(0.103)
    pink_forward.state = "push"
    robot = np.zeros(15, dtype=np.float32)
    robot[:3] = [0.08, -0.05, 0.464]
    robot[3:6] = pink_forward.orientation
    object_pos = np.asarray([0.08, 0.05, 0.46], dtype=np.float32)
    post_hold = pink_forward.command(robot, object_pos=object_pos, oracle_success=False)
    assert np.isclose(object_pos[1] - post_hold["target_pos"][1], 0.015)


def test_projected_progress_anticipates_positive_motion_only(module):
    assert np.isclose(module.projected_push_progress(0.08, 0.07, horizon=3), 0.11)
    assert np.isclose(module.projected_push_progress(0.08, 0.09, horizon=3), 0.08)


def test_controller_advances_only_after_reaching_waypoint(module):
    pos = np.asarray([0.10, -0.15, 0.46], dtype=np.float32)
    expert = module.ParametricPushExpert("push_blue_block_left", pos)
    far_robot = np.zeros(15, dtype=np.float32)
    far_robot[3:6] = expert.orientation
    first = expert.command(far_robot, object_pos=pos, oracle_success=False)
    assert first["state"] == "hover"
    assert expert.state == "hover"
    at_hover = far_robot.copy(); at_hover[:3] = expert.waypoints["hover"]
    for _ in range(expert.stable_steps):
        expert.command(at_hover, object_pos=pos, oracle_success=False)
    assert expert.state == "contact"
    at_contact = far_robot.copy(); at_contact[:3] = expert.waypoints["contact"]
    for _ in range(expert.stable_steps):
        expert.command(at_contact, object_pos=pos, oracle_success=False)
    assert expert.state == "push"


def test_guidance_is_object_named_and_phase_grounded(module):
    start = np.asarray([0.1, -0.15, 0.46], dtype=np.float32)
    robot = np.zeros(15, dtype=np.float32); robot[:3] = [0.2, -0.15, 0.47]
    before = module.build_push_guidance("push_pink_block_right", robot, start, start, motion_threshold=0.003)
    moved = start + np.asarray([0.004, 0.0, 0.0], dtype=np.float32)
    after = module.build_push_guidance("push_pink_block_right", robot, start, moved, motion_threshold=0.003)
    assert before["phase"] == "approach"
    assert not before["post_contact"]
    assert before["g"][0] < 0.0
    np.testing.assert_allclose(before["g"][3:], [1.0, 0.0, 0.0])
    assert after["phase"] == "contact_transport"
    np.testing.assert_allclose(after["g"][:3], 0.0)


def test_checkerboard_folds_rotate_bridges_without_pair_leakage(module):
    records = [
        {"task": task, "oracle_success": True, "path": f"accepted/{task}.npz", "attempt_id": task}
        for task in module.PUSH_TASKS
    ] + [
        {"task": "push_red_block_left", "oracle_success": False, "path": "attempts/fail.npz", "attempt_id": "fail"}
    ]
    folds = module.build_checkerboard_folds(records)
    assert len(folds) == 6
    assert {fold["bridge_object"] for fold in folds} == {"red", "blue", "pink"}
    assert all(sum(fold["bridge_object"] == color for fold in folds) == 2 for color in module.COLORS)
    heldout_frequency = {task: 0 for task in module.PUSH_TASKS}
    for fold in folds:
        source = set(fold["source_cells"])
        heldout = set(fold["heldout_cells"])
        assert not source & heldout
        assert len(source) == 4 and len(heldout) == 2
        source_objects = {module.parse_push_task(task)["color"] for task in source}
        source_directions = {module.parse_push_task(task)["direction"] for task in source}
        for task in heldout:
            spec = module.parse_push_task(task)
            assert spec["color"] in source_objects
            assert spec["direction"] in source_directions
        selected = fold["source_records"] + fold["heldout_records"]
        assert all(row["oracle_success"] for row in selected)
        assert all(row["attempt_id"] != "fail" for row in selected)
        for task in heldout:
            heldout_frequency[task] += 1
    assert set(heldout_frequency.values()) == {2}


def test_relative_action_conversion_clips_and_wraps(module):
    robot = np.zeros(15, dtype=np.float32)
    robot[:3] = [0.0, 0.0, 0.5]
    robot[3:6] = [3.13, 0.0, 0.0]
    absolute = np.asarray([0.04, -0.01, 0.5, -3.13, 0.0, 0.0, -1.0], dtype=np.float32)
    rel = module.absolute_to_relative_action(absolute, robot)
    np.testing.assert_allclose(rel[:3], [1.0, -0.5, 0.0])
    assert abs(float(rel[3])) < 1.0
    assert rel[-1] == -1.0


def test_reset_bank_is_deterministic_and_task_exact(module):
    rows = []
    for index, task in enumerate(module.PUSH_TASKS):
        rows.append({"task": task, "scene": "calvin_scene_D", "start": 100 + index, "end": 110 + index})
        rows.append({"task": task, "scene": "calvin_scene_B", "start": 200 + index, "end": 210 + index})
    rows.append({"task": "open_drawer", "scene": "calvin_scene_D", "start": 999, "end": 1000})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "manifest.json"
        path.write_text(json.dumps({"segments_detail": rows}), encoding="utf-8")
        first = module.load_reset_bank(path, seed=7)
        second = module.load_reset_bank(path, seed=7)
    assert first == second
    assert set(first) == set(module.PUSH_TASKS)
    assert all(len(items) == 2 for items in first.values())
    assert all(item["task"] == task for task, items in first.items() for item in items)


def test_fixed_scene_reset_bank_reuses_every_reset_across_all_cells(module):
    resets = []
    for index in range(3):
        resets.append(
            {
                "reset_id": f"scene_D_native_{index:04d}",
                "scene": "calvin_scene_D",
                "runtime_scene": "calvin_scene_D",
                "source_kind": "native_runtime_reset",
                "robot_obs": np.zeros(15, dtype=np.float32).tolist(),
                "scene_obs": np.zeros(24, dtype=np.float32).tolist(),
                "eligible_tasks": list(module.PUSH_TASKS),
            }
        )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fixed_resets.json"
        path.write_text(
            json.dumps(
                {
                    "format": "calvin_fixed_scene_push_resets_v1",
                    "runtime_scene": "calvin_scene_D",
                    "resets": resets,
                }
            ),
            encoding="utf-8",
        )
        bank = module.load_reset_bank(path, seed=7)
    assert all(len(bank[task]) == 3 for task in module.PUSH_TASKS)
    expected_ids = [row["reset_id"] for row in bank[module.PUSH_TASKS[0]]]
    assert all([row["reset_id"] for row in bank[task]] == expected_ids for task in module.PUSH_TASKS)
    assert set(expected_ids) == {r["reset_id"] for r in resets}
    assert all(row["runtime_scene"] == "calvin_scene_D" for rows in bank.values() for row in rows)


def test_fixed_scene_reset_bank_supports_task_eligible_subsets(module):
    tasks = ("push_red_block_forward", "push_red_block_backward")
    reset = {
        "reset_id": "red_y_0000",
        "scene": "calvin_scene_D",
        "runtime_scene": "calvin_scene_D",
        "source_kind": "native_runtime_reset",
        "robot_obs": np.zeros(15, dtype=np.float32).tolist(),
        "scene_obs": np.zeros(24, dtype=np.float32).tolist(),
        "eligible_tasks": list(tasks),
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "y_resets.json"
        path.write_text(json.dumps({"format": "calvin_fixed_scene_push_resets_v1", "runtime_scene": "calvin_scene_D", "resets": [reset]}), encoding="utf-8")
        bank = module.load_reset_bank(path, seed=4, tasks=tasks)
    assert [row["reset_id"] for row in bank[tasks[0]]] == ["red_y_0000"]
    assert [row["reset_id"] for row in bank[tasks[1]]] == ["red_y_0000"]


def test_embedded_reset_arrays_do_not_read_cross_scene_episode(module):
    reset = {
        "robot_obs": np.arange(15, dtype=np.float32).tolist(),
        "scene_obs": np.arange(24, dtype=np.float32).tolist(),
    }
    robot, scene = module.load_reset_arrays(Path("/must/not/be/read"), reset)
    np.testing.assert_allclose(robot, np.arange(15, dtype=np.float32))
    np.testing.assert_allclose(scene, np.arange(24, dtype=np.float32))


def test_failure_trajectory_is_temporally_thinned_but_keeps_last_step(module):
    steps = 10
    trajectory = {
        "robot_obs": np.arange(steps * 15, dtype=np.float32).reshape(steps, 15),
        "scene_obs": np.zeros((steps, 24), dtype=np.float32),
        "rgb_static": np.zeros((steps, 2, 2, 3), dtype=np.uint8),
        "rgb_gripper": np.zeros((steps, 2, 2, 3), dtype=np.uint8),
        "actions": np.zeros((steps, 7), dtype=np.float32),
        "rel_actions": np.zeros((steps, 7), dtype=np.float32),
        "g": np.zeros((steps, 6), dtype=np.float32),
        "phase": np.asarray(["approach"] * steps),
        "post_contact": np.zeros(steps, dtype=bool),
        "controller_state": np.asarray(["hover"] * steps),
        "object_pos": np.zeros((steps, 3), dtype=np.float32),
        "interaction_anchor": np.zeros((steps, 3), dtype=np.float32),
        "progress": np.zeros(steps, dtype=np.float32),
        "task": np.asarray("push_red_block_forward"),
        "object_name": np.asarray("block_red"),
        "direction": np.asarray("forward"),
        "success": np.asarray(False),
    }
    thinned = module.lightweight_failure_trajectory(trajectory, stride=4)
    assert len(thinned["robot_obs"]) == 4
    np.testing.assert_array_equal(thinned["robot_obs"][-1], trajectory["robot_obs"][-1])
    assert int(np.asarray(thinned["original_steps"]).item()) == steps
    assert str(np.asarray(thinned["failure_storage_mode"]).item()) == "temporal_stride"
    module.validate_parametric_trajectory(thinned)


def test_parametric_schema_binds_task_to_factor_labels(module):
    steps = 2
    trajectory = {
        "robot_obs": np.zeros((steps, 15), dtype=np.float32),
        "scene_obs": np.zeros((steps, 24), dtype=np.float32),
        "rgb_static": np.zeros((steps, 200, 200, 3), dtype=np.uint8),
        "rgb_gripper": np.zeros((steps, 84, 84, 3), dtype=np.uint8),
        "actions": np.zeros((steps, 7), dtype=np.float32),
        "rel_actions": np.zeros((steps, 7), dtype=np.float32),
        "g": np.zeros((steps, 6), dtype=np.float32),
        "phase": np.asarray(["approach", "contact_transport"]),
        "post_contact": np.asarray([False, True]),
        "controller_state": np.asarray(["hover", "push"]),
        "object_pos": np.zeros((steps, 3), dtype=np.float32),
        "task": np.asarray("push_red_block_left"),
        "object_name": np.asarray("block_red"),
        "direction": np.asarray("left"),
        "success": np.asarray(True),
    }
    module.validate_parametric_trajectory(trajectory)
    trajectory["direction"] = np.asarray("right")
    try:
        module.validate_parametric_trajectory(trajectory)
    except ValueError as exc:
        assert "factor" in str(exc).lower()
    else:
        raise AssertionError("task/factor mismatch must be rejected")


def test_native_collection_contract_does_not_require_candidate_g(module):
    steps = 2
    trajectory = {
        "robot_obs": np.zeros((steps, 15), dtype=np.float32),
        "scene_obs": np.zeros((steps, 24), dtype=np.float32),
        "rgb_static": np.zeros((steps, 200, 200, 3), dtype=np.uint8),
        "rgb_gripper": np.zeros((steps, 84, 84, 3), dtype=np.uint8),
        "actions": np.zeros((steps, 7), dtype=np.float32),
        "rel_actions": np.zeros((steps, 7), dtype=np.float32),
        "controller_state": np.asarray(["hover", "push"]),
        "object_pos": np.zeros((steps, 3), dtype=np.float32),
        "progress": np.zeros(steps, dtype=np.float32),
        "task": np.asarray("push_red_block_forward"),
        "object_name": np.asarray("block_red"),
        "direction": np.asarray("forward"),
        "success": np.asarray(True),
        "provisional_label_schema": np.asarray("not_emitted"),
    }
    module.validate_parametric_trajectory(trajectory)
    partial = dict(trajectory)
    partial["g"] = np.zeros((steps, 6), dtype=np.float32)
    try:
        module.validate_parametric_trajectory(partial)
    except ValueError as exc:
        assert "provisional" in str(exc).lower()
    else:
        raise AssertionError("partial provisional labels must be rejected")


def test_metadata_only_failure_and_explicit_prune_sidecar(module):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        receipt_path = module.persist_failure_metadata_only(
            root,
            attempt_id="failed-0",
            receipt={"task": "push_red_block_forward", "oracle_success": False},
        )
        assert receipt_path == root / "failures/failed-0.receipt.json"
        assert receipt_path.is_file()
        assert not list(root.rglob("failed-0.npz"))

        accepted_trajectory = {
            "robot_obs": np.zeros((1, 15), dtype=np.float32),
            "scene_obs": np.zeros((1, 24), dtype=np.float32),
            "rgb_static": np.zeros((1, 2, 2, 3), dtype=np.uint8),
            "rgb_gripper": np.zeros((1, 2, 2, 3), dtype=np.uint8),
            "actions": np.zeros((1, 7), dtype=np.float32),
            "rel_actions": np.zeros((1, 7), dtype=np.float32),
            "controller_state": np.asarray(["push"]),
            "object_pos": np.zeros((1, 3), dtype=np.float32),
            "progress": np.zeros(1, dtype=np.float32),
            "task": np.asarray("push_red_block_forward"),
            "object_name": np.asarray("block_red"),
            "direction": np.asarray("forward"),
            "success": np.asarray(True),
            "provisional_label_schema": np.asarray("not_emitted"),
        }
        accepted = module.persist_attempt(
            root,
            attempt_id="accepted-0",
            trajectory=accepted_trajectory,
            receipt={"task": "push_red_block_forward", "oracle_success": True},
        )
        missing_failure = "attempts/old-failed.npz"
        manifest = {
            "format": "calvin_parametric_push_collection_v1",
            "tasks": ["push_red_block_forward"],
            "quota_per_cell": 1,
            "results": [
                {"task": "push_red_block_forward", "oracle_success": True, "path": accepted.relative_to(root).as_posix(), "sha256": module.sha256_file(accepted)},
                {"task": "push_red_block_forward", "oracle_success": False, "path": missing_failure, "sha256": "deleted"},
            ],
            "folds": [],
        }
        (root / "collection_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (root / "failure_payload_prune.json").write_text(
            json.dumps({"format": "calvin_failure_payload_prune_v1", "pruned_paths": [missing_failure]}),
            encoding="utf-8",
        )
        summary = module.validate_collection(root)
        assert summary["accepted_verified"] == 1
        assert summary["failed_payloads_pruned"] == 1
        accepted.unlink()
        try:
            module.validate_collection(root)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing accepted payload must fail closed")


def test_provisional_g_is_explicit_opt_in(module):
    native = {"actions": np.zeros(7, dtype=np.float32)}
    guidance = {
        "g": np.zeros(6, dtype=np.float32),
        "phase": "approach",
        "post_contact": False,
        "interaction_anchor": np.zeros(3, dtype=np.float32),
    }
    default_step = module.attach_provisional_labels(native, guidance, emit=False)
    assert set(default_step) == {"actions"}
    labeled_step = module.attach_provisional_labels(native, guidance, emit=True)
    assert {"g", "phase", "post_contact", "interaction_anchor"} <= set(labeled_step)


def main():
    module = load_module()
    test_task_parser_covers_full_factorial(module)
    test_controller_mirrors_contact_side_and_push_direction(module)
    test_parser_and_controller_cover_world_y_directions(module)
    test_y_controller_tracks_live_object_with_lateral_error_correction(module)
    test_y_controller_uses_direction_and_object_specific_settle_parameters(module)
    test_projected_progress_anticipates_positive_motion_only(module)
    test_controller_advances_only_after_reaching_waypoint(module)
    test_guidance_is_object_named_and_phase_grounded(module)
    test_checkerboard_folds_rotate_bridges_without_pair_leakage(module)
    test_relative_action_conversion_clips_and_wraps(module)
    test_reset_bank_is_deterministic_and_task_exact(module)
    test_fixed_scene_reset_bank_reuses_every_reset_across_all_cells(module)
    test_fixed_scene_reset_bank_supports_task_eligible_subsets(module)
    test_embedded_reset_arrays_do_not_read_cross_scene_episode(module)
    test_failure_trajectory_is_temporally_thinned_but_keeps_last_step(module)
    test_parametric_schema_binds_task_to_factor_labels(module)
    test_native_collection_contract_does_not_require_candidate_g(module)
    test_metadata_only_failure_and_explicit_prune_sidecar(module)
    test_provisional_g_is_explicit_opt_in(module)
    print("CALVIN parametric push expert unit contract: OK")


if __name__ == "__main__":
    main()
