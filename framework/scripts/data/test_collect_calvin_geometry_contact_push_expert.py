#!/usr/bin/env python3
"""TDD contract for the CALVIN geometry-contact push planner."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO / "framework/scripts/data/collect_calvin_geometry_contact_push_expert.py"


def load_module():
    spec = importlib.util.spec_from_file_location("collect_calvin_geometry_contact_push_expert", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contact_candidates_are_mirrored_by_geometry_not_object_label(module):
    aabb = (np.asarray([0.04, -0.08, 0.44]), np.asarray([0.12, 0.00, 0.48]))
    forward = module.generate_contact_candidates(
        object_name="block_red", aabb=aabb, flow=np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    )
    backward = module.generate_contact_candidates(
        object_name="block_red", aabb=aabb, flow=np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
    )
    blue = module.generate_contact_candidates(
        object_name="block_blue", aabb=aabb, flow=np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    )
    assert len(forward) == len(backward) == len(blue) == 16
    assert [c.geometry_signature for c in forward] == [c.geometry_signature for c in blue]
    for fwd, back in zip(forward, backward):
        assert np.isclose(fwd.contact_pos[0], back.contact_pos[0])
        assert np.isclose(fwd.contact_pos[2], back.contact_pos[2])
        assert fwd.contact_pos[1] < aabb[0][1]
        assert back.contact_pos[1] > aabb[1][1]
        np.testing.assert_allclose(fwd.flow, -back.flow)


def test_oriented_contact_frame_uses_face_center_not_world_aabb_corner(module):
    angle = np.deg2rad(45.0)
    quaternion = np.asarray([0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)])
    candidates = module.generate_oriented_contact_candidates(
        object_name="block_red",
        object_pos=np.asarray([0.08, -0.04, 0.46]),
        object_orn=quaternion,
        local_extents=np.asarray([0.08, 0.04, 0.04]),
        flow=np.asarray([0.0, 1.0, 0.0]),
    )
    assert len(candidates) == 16
    for candidate in candidates:
        assert float(np.dot(candidate.surface_normal, [0.0, -1.0, 0.0])) > 0.70
        radial = candidate.contact_pos[:2] - np.asarray([0.08, -0.04])
        assert float(np.dot(radial, candidate.surface_normal[:2])) > 0.0
        assert np.isclose(candidate.contact_pos[0], 0.08, atol=1e-6)
    assert len({round(float(c.orientation[2]), 6) for c in candidates}) == 4


def test_candidate_contact_height_and_standoff_come_from_aabb(module):
    aabb = (np.asarray([0.02, -0.06, 0.44]), np.asarray([0.10, -0.01, 0.48]))
    candidates = module.generate_contact_candidates(
        object_name="block_pink", aabb=aabb, flow=np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    )
    heights = sorted({round(float(c.contact_pos[2]), 6) for c in candidates})
    assert heights == [0.448, 0.454, 0.46, 0.466]
    assert all(np.isclose(c.contact_pos[1], -0.075) for c in candidates)


def test_candidate_controller_uses_contact_frame_and_pushes_along_flow(module):
    aabb = (np.asarray([0.04, -0.08, 0.44]), np.asarray([0.12, 0.00, 0.48]))
    candidate = module.generate_contact_candidates(
        object_name="block_red", aabb=aabb, flow=np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    )[0]
    controller = module.GeometryContactController(candidate, push_distance=0.12)
    np.testing.assert_allclose(controller.waypoints["contact"], candidate.contact_pos)
    np.testing.assert_allclose(
        controller.waypoints["push"], candidate.contact_pos + np.asarray([0.0, 0.12, 0.0], dtype=np.float32)
    )
    robot = np.zeros(15, dtype=np.float32)
    robot[:3] = controller.waypoints["hover"]
    robot[3:6] = candidate.orientation
    command = None
    for _ in range(controller.stable_steps):
        command = controller.command(robot)
    assert command is not None
    assert command["state"] == "contact"
    robot[:3] = np.asarray(command["target_position"], dtype=np.float32)
    contact_command = controller.command(robot, contact_detected=True)
    assert contact_command["state"] == "push"
    np.testing.assert_allclose(
        np.asarray(contact_command["target_position"]),
        candidate.contact_pos + candidate.flow * controller.push_distance,
        atol=1e-6,
    )
    moved_candidate = candidate._replace(contact_pos=candidate.contact_pos + np.asarray([0.01, 0.02, 0.0]))
    controller.update_live_contact_frame(moved_candidate)
    tracked_command = controller.command(robot, contact_detected=True)
    np.testing.assert_allclose(
        np.asarray(tracked_command["target_position"]),
        moved_candidate.contact_pos + candidate.flow * controller.push_distance,
        atol=1e-6,
    )
    assert controller.waypoints["push"][1] > controller.waypoints["contact"][1]



def test_preview_score_rewards_progress_and_rejects_lateral_or_fallen_motion(module):
    stable = module.score_preview(
        progress=0.105, lateral_drift=0.005, z_drop=0.0, tilt=0.05,
        support_contact=True, robot_object_contacts=8, oracle_success=True,
    )
    lateral = module.score_preview(
        progress=0.120, lateral_drift=0.20, z_drop=0.0, tilt=0.05,
        support_contact=True, robot_object_contacts=8, oracle_success=False,
    )
    fallen = module.score_preview(
        progress=0.150, lateral_drift=0.01, z_drop=0.40, tilt=0.8,
        support_contact=False, robot_object_contacts=8, oracle_success=False,
    )
    no_contact = module.score_preview(
        progress=0.020, lateral_drift=0.0, z_drop=0.0, tilt=0.0,
        support_contact=True, robot_object_contacts=0, oracle_success=False,
    )
    assert stable > lateral
    assert stable > fallen
    assert lateral > fallen
    assert no_contact < 0.0


def test_planner_selects_success_then_best_safe_score(module):
    trials = [
        {"candidate_id": "a", "oracle_success": False, "score": 0.2},
        {"candidate_id": "b", "oracle_success": True, "score": 1.1},
        {"candidate_id": "c", "oracle_success": True, "score": 1.4},
    ]
    assert module.select_candidate_trial(trials)["candidate_id"] == "c"
    failures = [
        {"candidate_id": "a", "oracle_success": False, "score": -0.2},
        {"candidate_id": "b", "oracle_success": False, "score": 0.1},
    ]
    assert module.select_candidate_trial(failures)["candidate_id"] == "b"


def test_probe_mode_selects_exact_reset_ids_and_repeats(module):
    resets = [
        {"reset_id": "red_0004", "value": 1},
        {"reset_id": "blue_0003", "value": 2},
        {"reset_id": "other", "value": 3},
    ]
    selected = module.select_probe_resets(resets, ("blue_0003", "red_0004"), repeats_per_reset=3)
    assert [(row[0]["reset_id"], row[1]) for row in selected] == [
        ("blue_0003", 0), ("blue_0003", 1), ("blue_0003", 2),
        ("red_0004", 0), ("red_0004", 1), ("red_0004", 2),
    ]
    try:
        module.select_probe_resets(resets, ("missing",), repeats_per_reset=1)
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("missing reset id must fail closed")


def test_probe_mode_selects_one_exact_candidate(module):
    aabb = (np.asarray([0.04, -0.08, 0.44]), np.asarray([0.12, 0.00, 0.48]))
    candidates = module.generate_contact_candidates(
        object_name="block_blue", aabb=aabb, flow=np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    )
    selected = module.select_probe_candidate(candidates, "h0.35_q1")
    assert selected.candidate_id == "h0.35_q1"
    try:
        module.select_probe_candidate(candidates, "h9.99_q9")
    except ValueError as exc:
        assert "h9.99_q9" in str(exc)
    else:
        raise AssertionError("missing candidate id must fail closed")


def test_deterministic_reset_clears_all_movable_velocities_in_two_passes(module):
    class FakeBullet:
        def __init__(self):
            self.calls = []

        def resetBaseVelocity(self, uid, linearVelocity, angularVelocity, physicsClientId):
            self.calls.append((uid, tuple(linearVelocity), tuple(angularVelocity), physicsClientId))

    class FakeEnv:
        cid = 7

        def __init__(self):
            self.p = FakeBullet()
            self.reset_calls = 0

        def reset(self, robot_obs=None, scene_obs=None):
            self.reset_calls += 1
            return {"reset_call": self.reset_calls}

        def get_info(self):
            return {
                "scene_info": {
                    "movable_objects": {
                        "block_red": {"uid": 11},
                        "block_blue": {"uid": 12},
                    }
                }
            }

        def get_obs(self):
            return {"reset_call": self.reset_calls, "quiesced": True}

    env = FakeEnv()
    obs = module.deterministic_native_reset(env, np.zeros(15), np.zeros(24))
    assert env.reset_calls == 2
    assert obs["quiesced"]
    assert env.p.calls == [
        (11, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 7),
        (12, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 7),
        (11, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 7),
        (12, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 7),
    ]


def main():
    module = load_module()
    test_contact_candidates_are_mirrored_by_geometry_not_object_label(module)
    test_oriented_contact_frame_uses_face_center_not_world_aabb_corner(module)
    test_candidate_contact_height_and_standoff_come_from_aabb(module)
    test_candidate_controller_uses_contact_frame_and_pushes_along_flow(module)

    test_preview_score_rewards_progress_and_rejects_lateral_or_fallen_motion(module)
    test_planner_selects_success_then_best_safe_score(module)
    test_probe_mode_selects_exact_reset_ids_and_repeats(module)
    test_probe_mode_selects_one_exact_candidate(module)
    test_deterministic_reset_clears_all_movable_velocities_in_two_passes(module)
    print("CALVIN geometry-contact push planner unit contract: OK")


if __name__ == "__main__":
    main()
