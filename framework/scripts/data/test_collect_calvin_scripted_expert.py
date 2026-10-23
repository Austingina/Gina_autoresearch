#!/usr/bin/env python3
"""Unit contract for success-gated CALVIN scripted-expert collection."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np

REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO / "framework/scripts/data/collect_calvin_scripted_expert.py"


def load_module():
    spec = importlib.util.spec_from_file_location("collect_calvin_scripted_expert", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_trajectory(module, *, success: bool):
    steps = 3
    return {
        "robot_obs": np.zeros((steps, 15), dtype=np.float32),
        "scene_obs": np.zeros((steps, 24), dtype=np.float32),
        "rgb_static": np.zeros((steps, 200, 200, 3), dtype=np.uint8),
        "rgb_gripper": np.zeros((steps, 84, 84, 3), dtype=np.uint8),
        "actions": np.zeros((steps, 7), dtype=np.float32),
        "g": np.zeros((steps, 6), dtype=np.float32),
        "phase": np.asarray(["approach", "contact_transport", "contact_transport"]),
        "post_contact": np.asarray([False, True, True]),
        "task": np.asarray("push_blue_block_left"),
        "success": np.asarray(success),
    }


def test_guidance_switches_after_state_grounded_motion(module):
    initial_scene = np.zeros(24, dtype=np.float32)
    initial_scene[12:15] = [0.1, -0.2, 0.46]
    robot = np.zeros(15, dtype=np.float32)
    robot[:3] = [0.2, -0.2, 0.46]
    anchor = np.asarray([0.13, -0.2, 0.46], dtype=np.float32)

    before = module.build_guidance(
        "push_blue_block_left", robot, initial_scene, initial_scene.copy(), anchor, motion_threshold=0.005
    )
    moved_scene = initial_scene.copy()
    moved_scene[12] -= 0.006
    after = module.build_guidance(
        "push_blue_block_left", robot, initial_scene, moved_scene, anchor, motion_threshold=0.005
    )

    np.testing.assert_allclose(before["g"][:3], anchor - robot[:3])
    np.testing.assert_allclose(before["g"][3:], [-1.0, 0.0, 0.0])
    assert before["phase"] == "approach"
    assert not before["post_contact"]
    np.testing.assert_allclose(after["g"][:3], 0.0)
    np.testing.assert_allclose(after["g"][3:], [-1.0, 0.0, 0.0])
    assert after["phase"] == "contact_transport"
    assert after["post_contact"]


def test_articulation_progress_uses_joint_state(module):
    initial_scene = np.zeros(24, dtype=np.float32)
    drawer_scene = initial_scene.copy(); drawer_scene[1] = 0.02
    slider_scene = initial_scene.copy(); slider_scene[0] = 0.03
    np.testing.assert_allclose(module.task_progress("open_drawer", initial_scene, drawer_scene), 0.02)
    np.testing.assert_allclose(module.task_progress("move_slider_left", initial_scene, slider_scene), 0.03)


def test_success_gate_separates_accepted_and_failed(module):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        accepted = module.persist_attempt(
            root,
            attempt_id="push-ok",
            trajectory=fake_trajectory(module, success=True),
            receipt={"task": "push_blue_block_left", "oracle_success": True},
        )
        failed = module.persist_attempt(
            root,
            attempt_id="push-fail",
            trajectory=fake_trajectory(module, success=False),
            receipt={"task": "push_blue_block_left", "oracle_success": False},
        )
        assert accepted.parent.name == "accepted"
        assert failed.parent.name == "attempts"
        assert (root / "accepted/push-ok.npz").is_file()
        assert not (root / "accepted/push-fail.npz").exists()
        assert (root / "attempts/push-fail.npz").is_file()


def test_schema_validator_rejects_misaligned_modalities(module):
    trajectory = fake_trajectory(module, success=True)
    trajectory["actions"] = trajectory["actions"][:2]
    try:
        module.validate_trajectory(trajectory)
    except ValueError as exc:
        assert "aligned" in str(exc)
    else:
        raise AssertionError("misaligned trajectory must be rejected")


def test_collection_readback_verifies_hash_and_schema(module):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = module.persist_attempt(
            root,
            attempt_id="push-ok",
            trajectory=fake_trajectory(module, success=True),
            receipt={"task": "push_blue_block_left", "oracle_success": True},
        )
        manifest = {
            "format": "calvin_scripted_expert_collection_v1",
            "accepted": 1,
            "failed": 0,
            "results": [
                {
                    "task": "push_blue_block_left",
                    "oracle_success": True,
                    "path": path.relative_to(root).as_posix(),
                    "sha256": module.sha256_file(path),
                }
            ],
        }
        (root / "collection_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        summary = module.validate_collection(root)
        assert summary["accepted_verified"] == 1
        assert summary["tasks"] == ["push_blue_block_left"]


def test_native_contract_omits_candidate_g(module):
    trajectory = fake_trajectory(module, success=True)
    for name in module.PROVISIONAL_STEP_FIELDS:
        trajectory.pop(name)
    trajectory["provisional_label_schema"] = np.asarray("not_emitted")
    module.validate_trajectory(trajectory)


def test_metadata_only_failure_writes_no_npz(module):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        receipt_path = module.persist_failure_metadata_only(
            root,
            attempt_id="failed-native",
            receipt={"task": "push_blue_block_left", "oracle_success": False},
        )
        assert receipt_path.is_file()
        assert not list(root.rglob("*.npz"))


def main():
    module = load_module()
    test_guidance_switches_after_state_grounded_motion(module)
    test_articulation_progress_uses_joint_state(module)
    test_success_gate_separates_accepted_and_failed(module)
    test_schema_validator_rejects_misaligned_modalities(module)
    test_collection_readback_verifies_hash_and_schema(module)
    test_native_contract_omits_candidate_g(module)
    test_metadata_only_failure_writes_no_npz(module)
    print("CALVIN scripted expert collector unit contract: OK")


if __name__ == "__main__":
    main()
