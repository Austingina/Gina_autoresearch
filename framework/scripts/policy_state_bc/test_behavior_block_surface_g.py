#!/usr/bin/env python3
"""Small geometry regression test for the behavior block-surface G builder."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, Path(__file__).resolve().parent.as_posix())

from behavior_surface_geometry import (  # noqa: E402
    oriented_box_ray_distance,
    oriented_box_required_clearance,
    oriented_box_volume_delta,
)


def load_policy_module():
    try:
        import train_eval_state_chunk_flow_policy as policy
    except ModuleNotFoundError as error:
        if error.name == "torch":
            return None
        raise
    return policy


def main() -> None:
    half_extents = np.asarray([0.02, 0.03, 0.04], dtype=np.float32)
    flow = np.asarray([-1.0, 0.0, 0.0], dtype=np.float32)
    distance_yaw0 = oriented_box_ray_distance(
        flow, np.asarray([0.0, 0.0, 0.0], dtype=np.float32), half_extents
    )
    distance_yaw90 = oriented_box_ray_distance(
        flow, np.asarray([0.0, 0.0, np.pi / 2], dtype=np.float32), half_extents
    )
    assert np.isclose(distance_yaw0, 0.02), distance_yaw0
    assert np.isclose(distance_yaw90, 0.03), distance_yaw90
    point = np.asarray([0.05, 0.00, 0.00], dtype=np.float32)
    center = np.zeros(3, dtype=np.float32)
    euler = np.zeros(3, dtype=np.float32)
    delta = oriented_box_volume_delta(point, center, euler, half_extents)
    assert np.allclose(delta, np.asarray([-0.03, 0.0, 0.0])), delta
    assert np.isclose(
        oriented_box_required_clearance(point, center, euler, half_extents), 0.03
    )
    expanded_delta = oriented_box_volume_delta(
        point, center, euler, half_extents, clearance=0.03
    )
    assert np.allclose(expanded_delta, 0), expanded_delta

    policy = load_policy_module()
    if policy is not None:
        import torch

        torch.manual_seed(0)
        factorized = policy.FactorizedMinimalGTokenChunkPolicy(
            base_dim=15,
            g_dim=6,
            chunk_size=4,
            hidden=32,
            layers=1,
        ).eval()
        base_input = torch.randn(3, 15)
        g_input = torch.randn(3, 6)
        with torch.inference_mode():
            factorized_normal = factorized(base_input, g_input).reshape(3, 4, 7)
            factorized_flip = factorized(base_input, -g_input).reshape(3, 4, 7)
        assert not torch.allclose(
            factorized_normal[:, :, :3], factorized_flip[:, :, :3]
        )
        assert torch.equal(
            factorized_normal[:, :, 3:7], factorized_flip[:, :, 3:7]
        )

        assert not policy.behavior_object_motion_started(
            np.zeros(3), np.asarray([0.0019, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0]), 0.002
        )
        assert policy.behavior_object_motion_started(
            np.zeros(3), np.asarray([0.0021, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0]), 0.002
        )
        assert not policy.behavior_object_motion_started(
            np.zeros(3), np.asarray([0.0030, 0.0, 0.0]), np.asarray([-1.0, 0.0, 0.0]), 0.002
        )

        robot = np.zeros((1, 15), dtype=np.float32)
        scene = np.zeros((1, 24), dtype=np.float32)
        scene[0, 6:9] = np.asarray([0.10, 0.0, 0.0], dtype=np.float32)
        task = np.asarray(["push_red_block_right"], dtype=object)
        kwargs = {
            "block_half_extents": {"red": np.full(3, 0.02, dtype=np.float32)},
            "block_scene_slices": {"red": slice(6, 9)},
            "tool_clearance": 0.014,
            "contact_tolerance": 0.005,
        }
        pre = policy.make_behavior_block_surface_minimal_g(
            robot, scene, task, post_contact=np.asarray([False]), **kwargs
        )
        post = policy.make_behavior_block_surface_minimal_g(
            robot, scene, task, post_contact=np.asarray([True]), **kwargs
        )
        assert np.linalg.norm(pre[0, :3]) > 0
        assert np.allclose(post[0, :3], 0)
        assert np.allclose(post[0, 3:6], np.asarray([1.0, 0.0, 0.0]))

        volume = policy.make_behavior_block_volume_minimal_g(
            robot,
            scene,
            task,
            block_half_extents=kwargs["block_half_extents"],
            block_scene_slices=kwargs["block_scene_slices"],
            tool_clearance=0.018,
            post_contact=np.asarray([False]),
        )
        assert np.allclose(volume[0, :3], np.asarray([0.062, 0.0, 0.0]))
        assert np.allclose(volume[0, 3:6], np.asarray([1.0, 0.0, 0.0]))

        state, language = policy.make_state_lang_parts(
            {
                "robot_obs": np.arange(30, dtype=np.float32).reshape(2, 15),
                "scene_obs": np.zeros((2, 24), dtype=np.float32),
                "task_oh": np.zeros((2, 3), dtype=np.float32),
            },
            state_scope="proprio_no_eef_xyz",
            language_mode="none",
        )
        assert state.shape == (2, 12)
        assert np.array_equal(
            state,
            np.arange(30, dtype=np.float32).reshape(2, 15)[:, 3:],
        )
        assert language.shape == (2, 0)

        relative_state, language = policy.make_state_lang_parts(
            {
                "robot_obs": np.arange(30, dtype=np.float32).reshape(2, 15),
                "scene_obs": np.zeros((2, 24), dtype=np.float32),
                "task_oh": np.zeros((2, 3), dtype=np.float32),
            },
            state_scope="eef_orientation_gripper",
            language_mode="none",
        )
        expected = np.concatenate(
            [
                np.arange(30, dtype=np.float32).reshape(2, 15)[:, 3:7],
                np.arange(30, dtype=np.float32).reshape(2, 15)[:, 14:15],
            ],
            axis=1,
        )
        assert relative_state.shape == (2, 5)
        assert np.array_equal(relative_state, expected)
        assert language.shape == (2, 0)

        pair_robot = np.zeros((4, 15), dtype=np.float32)
        pair_robot[1, 0] = 0.005
        pair_robot[3, 0] = 0.005
        pair_g = np.asarray(
            [
                [0.10, 0.00, 0.00, 1.0, 0.0, 0.0],
                [0.08, 0.00, 0.00, 1.0, 0.0, 0.0],
                [0.00, 0.10, 0.00, 1.0, 0.0, 0.0],
                [0.00, 0.08, 0.00, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        pair_actions = np.zeros((4, 1, 7), dtype=np.float32)
        pair_actions[:2, 0, 0] = 0.5
        pair_actions[2:, 0, 1] = 0.5
        pairs = policy.build_same_flow_target_train_pairs(
            {
                "task_name": np.asarray(["target_a", "target_a", "target_b", "target_b"]),
                "robot_obs": pair_robot,
                "articulation_minimal_g": pair_g,
                "action_chunks": pair_actions,
                "segment_id": np.asarray([0, 0, 1, 1]),
                "frame_index": np.asarray([0, 1, 0, 1]),
            },
            np.arange(4),
            max_relative_frame=32,
            min_attraction_norm=0.015,
            min_attraction_difference=0.02,
            min_action_difference=0.03,
            max_eef_distance_m=0.05,
            max_orientation_distance=0.6,
            own_alignment_threshold=0.2,
        )
        assert set(pairs) == {"target_a__target_b"}
        assert pairs["target_a__target_b"].shape == (2, 2)

        action_difference = torch.zeros((2, 14), dtype=torch.float32)
        action_difference[:, 3:7] = 1.0
        action_difference[:, 7:10] = 2.0
        assert torch.count_nonzero(
            policy.select_pair_action_scope(
                action_difference, chunk_size=2, scope="first_xyz"
            )
        ) == 0
        assert torch.equal(
            policy.select_pair_action_scope(
                action_difference, chunk_size=2, scope="all"
            ),
            action_difference,
        )

    print("behavior block-surface G: OK")


if __name__ == "__main__":
    main()
