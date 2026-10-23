#!/usr/bin/env python3
"""Pure-logic contract tests for the pinned LIBERO middle-drawer expert."""
from __future__ import annotations

import inspect
import unittest

import numpy as np

from framework.sim_data import libero_middle_drawer_expert as expert


class FakeModel:
    """Small MuJoCo-shaped model; no LIBERO or MuJoCo dependency."""

    def __init__(
        self,
        *,
        joint_types=(2,),
        joint_ranges=((-0.16, 0.01),),
        joint_bodies=(1,),
        joint_axes=((0.0, 1.0, 0.0),),
        nbody=2,
        body_names=("world", "cabinet_middle"),
    ):
        self.njnt = len(joint_types)
        self.nbody = nbody
        self.jnt_type = np.asarray(joint_types, dtype=np.int32)
        self.jnt_range = np.asarray(joint_ranges, dtype=np.float64)
        self.jnt_bodyid = np.asarray(joint_bodies, dtype=np.int32)
        self.jnt_axis = np.asarray(joint_axes, dtype=np.float64)
        self.body_names = tuple(body_names)


class FakeData:
    def __init__(self, *, body_position=(1.0, 2.0, 3.0), body_rotation=None, nbody=2):
        self.xpos = np.zeros((nbody, 3), dtype=np.float64)
        self.xpos[1] = np.asarray(body_position, dtype=np.float64)
        self.xmat = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], nbody, axis=0)
        if body_rotation is not None:
            self.xmat[1] = np.asarray(body_rotation, dtype=np.float64)


def valid_geometry():
    return expert.resolve_middle_drawer_geometry(FakeModel(), FakeData(), candidate_joint_ids=(0,))


class ActionContractTests(unittest.TestCase):
    def test_action_is_exactly_seven_finite_clipped_values(self):
        action = expert.sanitize_action([0.0, -4.0, -0.5, 0.0, 0.5, 4.0, 0.0])
        self.assertEqual(action.shape, (7,))
        self.assertTrue(np.isfinite(action).all())
        np.testing.assert_allclose(action, [0.0, -1.0, -0.5, 0.0, 0.5, 1.0, 0.0])
        with self.assertRaises(ValueError):
            expert.sanitize_action([np.inf, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan])
        with self.assertRaises(ValueError):
            expert.sanitize_action([0.0] * 6)


class GeometryContractTests(unittest.TestCase):
    def test_pinned_handle_offset_matches_upstream_xml(self):
        np.testing.assert_allclose(expert.MIDDLE_HANDLE_LOCAL, [-0.00278, -0.10161, 0.11041])

    def test_parser_requires_exactly_one_explicit_target_joint(self):
        with self.assertRaises(expert.GeometryError):
            expert.resolve_middle_drawer_geometry(
                FakeModel(), FakeData(), candidate_joint_ids=()
            )
        with self.assertRaises(expert.GeometryError):
            expert.resolve_middle_drawer_geometry(
                FakeModel(
                    joint_types=(2, 2),
                    joint_ranges=((-0.16, 0.01), (-0.16, 0.01)),
                    joint_bodies=(1, 2),
                    joint_axes=((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
                    nbody=3,
                    body_names=("world", "cabinet_middle", "other_drawer"),
                ),
                FakeData(nbody=3),
                candidate_joint_ids=(0, 1),
            )

        # Other slide joints may exist; the adapter must nominate exactly the
        # BDDL middle-region joint rather than imposing global uniqueness.
        geometry = expert.resolve_middle_drawer_geometry(
            FakeModel(
                joint_types=(2, 2),
                joint_ranges=((-0.16, 0.01), (-0.5, 0.5)),
                joint_bodies=(1, 2),
                joint_axes=((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
                nbody=3,
                body_names=("world", "cabinet_middle", "other_drawer"),
            ),
            FakeData(nbody=3),
            candidate_joint_ids=(0,),
        )
        self.assertEqual(geometry.joint_id, 0)

    def test_parser_rejects_bad_range_body_and_axis(self):
        invalid_models = (
            FakeModel(joint_types=(0,)),
            FakeModel(joint_ranges=((-0.30, 0.0),)),
            FakeModel(joint_ranges=((-np.inf, 0.0),)),
            FakeModel(joint_bodies=(2,)),
            FakeModel(joint_axes=((0.0, 0.0, 0.0),)),
            FakeModel(body_names=("world", "wrong_body")),
        )
        for model in invalid_models:
            with self.subTest(model=model), self.assertRaises(expert.GeometryError):
                expert.resolve_middle_drawer_geometry(model, FakeData(), candidate_joint_ids=(0,))

    def test_middle_handle_local_is_transformed_from_body_frame_to_world(self):
        rotation = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
        data = FakeData(body_position=(1.0, 2.0, 3.0), body_rotation=rotation)
        geometry = expert.resolve_middle_drawer_geometry(FakeModel(), data, candidate_joint_ids=(0,))
        expected = data.xpos[1] + rotation @ expert.MIDDLE_HANDLE_LOCAL
        np.testing.assert_allclose(geometry.handle_world, expected)

    def test_pull_direction_has_strictly_negative_projected_joint_delta(self):
        rotation = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
        geometry = expert.resolve_middle_drawer_geometry(
            FakeModel(joint_axes=((1.0, 0.0, 0.0),)),
            FakeData(body_rotation=rotation),
            candidate_joint_ids=(0,),
        )
        self.assertLess(float(np.dot(geometry.pull_direction, geometry.axis_world)), 0.0)
        self.assertLess(expert.projected_joint_delta(geometry, 0.08), 0.0)

    def test_missing_or_nan_geometry_fails_closed(self):
        data = FakeData()
        data.xpos[1, 0] = np.nan
        with self.assertRaises(expert.GeometryError):
            expert.resolve_middle_drawer_geometry(FakeModel(), data, candidate_joint_ids=(0,))
        with self.assertRaises(expert.GeometryError):
            expert.resolve_middle_drawer_geometry(object(), object(), candidate_joint_ids=(0,))


class FSMContractTests(unittest.TestCase):
    def test_fsm_reaches_hold_only_after_consecutive_error_confirmations(self):
        geometry = valid_geometry()
        controller = expert.MiddleDrawerExpert(confirm_steps=2, position_tolerance=1e-6)

        def confirm_position(expected_state, position):
            first = controller.step(position, geometry, gripper_error=1.0)
            self.assertEqual(first.state, expected_state)
            second = controller.step(position, geometry, gripper_error=1.0)
            return second

        pregrasp = geometry.handle_world + geometry.pull_direction * controller.pregrasp_offset
        self.assertEqual(controller.step([0.0, 0.0, 0.0], geometry).action[-1], -1.0)
        transition = confirm_position(expert.State.PREGRASP, pregrasp)
        self.assertEqual(transition.state, expert.State.APPROACH)

        # One bad positional frame resets confirmation instead of advancing.
        controller.step(geometry.handle_world, geometry, gripper_error=1.0)
        reset = controller.step(geometry.handle_world + [0.1, 0.0, 0.0], geometry, gripper_error=1.0)
        self.assertEqual(reset.confirm_count, 0)
        transition = confirm_position(expert.State.APPROACH, geometry.handle_world)
        self.assertEqual(transition.state, expert.State.CLOSE_GRIPPER)

        close_one = controller.step(geometry.handle_world, geometry, gripper_error=0.0)
        self.assertEqual(close_one.state, expert.State.CLOSE_GRIPPER)
        close_reset = controller.step(geometry.handle_world, geometry, gripper_error=0.5)
        self.assertEqual(close_reset.confirm_count, 0)
        controller.step(geometry.handle_world, geometry, gripper_error=0.0)
        transition = controller.step(geometry.handle_world, geometry, gripper_error=0.0)
        self.assertEqual(transition.state, expert.State.PULL)
        self.assertEqual(transition.action[-1], 1.0)

        pull_target = geometry.handle_world + geometry.pull_direction * controller.pull_distance
        moved_geometry = expert.DrawerGeometry(
            joint_id=geometry.joint_id,
            body_id=geometry.body_id,
            joint_range=geometry.joint_range,
            axis_world=geometry.axis_world,
            handle_world=geometry.handle_world + [0.0, -0.05, 0.0],
            pull_direction=geometry.pull_direction,
        )
        # Pull target is latched at grasp closure and must not chase the moving drawer handle.
        np.testing.assert_allclose(controller.step(pull_target, moved_geometry).action[:3], 0.0)
        # Reset the one-frame confirmation consumed by the latch check above.
        controller.step(pull_target + [0.1, 0.0, 0.0], moved_geometry)
        transition = confirm_position(expert.State.PULL, pull_target)
        self.assertEqual(transition.state, expert.State.HOLD)
        hold = controller.step(pull_target, geometry, gripper_error=0.0)
        self.assertEqual(hold.state, expert.State.HOLD)
        self.assertEqual(hold.action.shape, (7,))
        self.assertTrue(np.isfinite(hold.action).all())

    def test_nan_observation_or_missing_geometry_enters_safe_failed_state(self):
        controller = expert.MiddleDrawerExpert(confirm_steps=2)
        failed = controller.step([np.nan, 0.0, 0.0], valid_geometry(), gripper_error=0.0)
        self.assertEqual(failed.state, expert.State.FAILED)
        self.assertTrue(failed.failed)
        np.testing.assert_allclose(failed.action, np.zeros(7))

        other = expert.MiddleDrawerExpert(confirm_steps=2)
        missing = other.step([0.0, 0.0, 0.0], None, gripper_error=0.0)
        self.assertEqual(missing.state, expert.State.FAILED)
        np.testing.assert_allclose(missing.action, np.zeros(7))

    def test_step_contract_has_no_success_label_or_forbidden_inputs(self):
        parameters = set(inspect.signature(expert.MiddleDrawerExpert.step).parameters)
        self.assertTrue(parameters.isdisjoint({"drawer_qpos", "qpos", "reward", "done", "G", "g"}))
        result = expert.MiddleDrawerExpert().step([0.0, 0.0, 0.0], valid_geometry())
        self.assertNotIn("success", result.__dataclass_fields__)


if __name__ == "__main__":
    unittest.main()
