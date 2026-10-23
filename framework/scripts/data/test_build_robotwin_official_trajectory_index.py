#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).with_name("build_robotwin_official_trajectory_index.py")
spec = importlib.util.spec_from_file_location("robotwin_index", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class PartitionPhasesTest(unittest.TestCase):
    def test_rejects_noncontiguous_or_empty_phase_partitions_before_index_write(self) -> None:
        invalid = [module._phase("approach", 0, 0, "test"), module._phase("transport", 1, 0, "test")]
        with self.assertRaisesRegex(ValueError, "invalid phase partition"):
            module.validate_phase_partition(invalid, 2)

    def test_immediate_gripper_release_never_emits_empty_transport_phase(self) -> None:
        n = 12
        left_gripper = np.array([1.0, 1.0, 1.0, 0.7, 0.1, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        handle = {
            "action/left_ee_poses": np.zeros((n, 7)),
            "action/right_ee_poses": np.zeros((n, 7)),
            "action/left_ee_joint_states": left_gripper.reshape(n, 1),
            "action/right_ee_joint_states": np.ones((n, 1)),
        }

        phases = module.partition_phases(handle, "pick_place", n)

        self.assertTrue(phases)
        self.assertTrue(all(phase["start"] <= phase["end"] for phase in phases), phases)
        self.assertEqual(list(range(n)), [step for phase in phases for step in range(phase["start"], phase["end"] + 1)])

    def test_late_gripper_close_never_emits_empty_fallback_transport_phase(self) -> None:
        n = 20
        left_gripper = np.array([1.0] * 16 + [0.7, 0.1, 0.7, 1.0])
        handle = {
            "action/left_ee_poses": np.zeros((n, 7)),
            "action/right_ee_poses": np.zeros((n, 7)),
            "action/left_ee_joint_states": left_gripper.reshape(n, 1),
            "action/right_ee_joint_states": np.ones((n, 1)),
        }

        phases = module.partition_phases(handle, "pick_place", n)

        self.assertTrue(all(phase["start"] <= phase["end"] for phase in phases), phases)
        self.assertEqual(list(range(n)), [step for phase in phases for step in range(phase["start"], phase["end"] + 1)])


if __name__ == "__main__":
    unittest.main()
