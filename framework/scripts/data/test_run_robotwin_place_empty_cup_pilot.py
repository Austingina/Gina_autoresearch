#!/usr/bin/env python3
"""Quality-gate regression tests for the RoboTwin cup pilot launcher."""
from __future__ import annotations

import math
import unittest

import numpy as np

from framework.scripts.data.run_robotwin_place_empty_cup_pilot import (
    _capture_upright_reference,
    _quality_probe,
)


class _Point:
    def __init__(self, p):
        self.p = np.asarray(p, dtype=float)


class _Pose:
    def __init__(self, rotation):
        self._matrix = np.eye(4)
        self._matrix[:3, :3] = np.asarray(rotation, dtype=float)

    def to_transformation_matrix(self):
        return self._matrix


class _Body:
    linear_velocity = np.zeros(3)
    angular_velocity = np.zeros(3)


class _Actor:
    def get_components(self):
        return [_Body()]


class _Wrapper:
    def __init__(self, name, point, rotation):
        self._name = name
        self._point = _Point(point)
        self._pose = _Pose(rotation)
        self.actor = _Actor()

    def get_functional_point(self, _index, _kind="pose"):
        return self._point

    def get_pose(self):
        return self._pose

    def get_name(self):
        return self._name


class _Task:
    def __init__(self, rotation):
        self.cup = _Wrapper("cup", [0.0, 0.0, 0.0], rotation)
        self.coaster = _Wrapper("coaster", [0.0, 0.0, 0.0], np.eye(3))

    def check_actors_contact(self, _cup, _coaster):
        return True


class UprightReferenceTests(unittest.TestCase):
    def test_tracks_setup_time_dominant_axis_instead_of_assuming_local_z(self):
        # Quaternion [0.5, 0.5, 0.5, 0.5] used by place_empty_cup maps
        # local Y to world Z; local Z is horizontal.
        initial = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        task = _Task(initial)
        _capture_upright_reference(task)
        self.assertEqual(task._gina_upright_reference["local_axis_index"], 1)
        self.assertAlmostEqual(task._gina_upright_reference["initial_tilt_deg"], 0.0)

        angle = math.radians(5.0)
        rotate_world_x = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(angle), -math.sin(angle)],
                [0.0, math.sin(angle), math.cos(angle)],
            ]
        )
        task.cup._pose = _Pose(rotate_world_x @ initial)
        quality = _quality_probe(task)

        self.assertAlmostEqual(quality["measurements"]["tilt_deg"], 5.0, places=6)
        self.assertTrue(quality["checks"]["tilt"])
        self.assertTrue(quality["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
