#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
from PIL import Image

MODULE_PATH = Path(__file__).with_name("build_vima_official_trajectory_index.py")
spec = importlib.util.spec_from_file_location("build_vima_official_trajectory_index", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(module)


class VimaOfficialIndexTests(unittest.TestCase):
    def make_parent(self, *, success: bool = True, steps: int = 4) -> tuple[Path, tempfile.TemporaryDirectory]:
        temp = tempfile.TemporaryDirectory()
        parent = Path(temp.name) / "rotate" / "000001"
        (parent / "rgb_front").mkdir(parents=True)
        (parent / "rgb_top").mkdir(parents=True)
        action = {
            "pose0_position": np.zeros((steps, 3), dtype=np.float32),
            "pose0_rotation": np.zeros((steps, 4), dtype=np.float32),
            "pose1_position": np.zeros((steps, 3), dtype=np.float32),
            "pose1_rotation": np.zeros((steps, 4), dtype=np.float32),
        }
        obs = {
            "segm": {
                "front": np.zeros((steps + 1, 8, 8), dtype=np.uint8),
                "top": np.zeros((steps + 1, 8, 8), dtype=np.uint8),
            },
            "ee": np.zeros((steps + 1,), dtype=np.int64),
        }
        trajectory = {
            "prompt": "Rotate {object} by 90 degrees.",
            "steps": steps,
            "success": success,
            "failure": not success,
        }
        for name, value in (("action.pkl", action), ("obs.pkl", obs), ("trajectory.pkl", trajectory)):
            (parent / name).write_bytes(pickle.dumps(value, protocol=4))
        for view in ("rgb_front", "rgb_top"):
            for frame in range(steps + 1):
                Image.new("RGB", (8, 8), color=(frame, 0, 0)).save(parent / view / f"{frame}.jpg")
        return parent, temp

    def test_inspects_successful_rgb_segmentation_parent(self) -> None:
        parent, temp = self.make_parent()
        self.addCleanup(temp.cleanup)
        record = module.inspect_parent(
            parent,
            task="rotate",
            trajectory_id="000001",
            source_revision="revision",
            payload_root=parent.parents[1],
        )
        self.assertEqual("vima:rotate:000001", record["parent_episode_id"])
        self.assertEqual("vima_bench", record["simulator"])
        self.assertEqual("rotate", record["native_task_id"])
        self.assertEqual("orientation_manipulation", record["canonical_task"]["family"])
        self.assertEqual("official_trajectory_reorganization_and_label_enrichment", record["provenance"]["class"])
        self.assertEqual(4, record["native_transition_count"])
        self.assertEqual(4, record["native_payload"]["transitions"])
        self.assertEqual(5, record["native_payload"]["observation_frames"])
        self.assertEqual(["front", "top"], record["native_payload"]["rgb_views"])
        self.assertTrue(record["parent_success_evidence"]["success"])
        self.assertEqual("orientation_manipulation", record["task_family"])
        phases = record["derived_phase_views"]
        self.assertEqual(0, phases[0]["start_transition"])
        self.assertEqual(4, phases[-1]["end_transition_exclusive"])
        self.assertTrue(all(not phase["independent_success"] for phase in phases))

    def test_rejects_native_failure(self) -> None:
        parent, temp = self.make_parent(success=False)
        self.addCleanup(temp.cleanup)
        with self.assertRaisesRegex(ValueError, "success"):
            module.inspect_parent(parent, task="rotate", trajectory_id="000001", source_revision="revision", payload_root=parent.parents[1])

    def test_short_parent_has_nonempty_exact_phase_coverage(self) -> None:
        phases = module.phase_views(2)
        self.assertEqual([(0, 1), (1, 2)], [(row["start_transition"], row["end_transition_exclusive"]) for row in phases])

    def test_mixed_candidates_exclude_bad_parent_without_rejecting_task(self) -> None:
        good, good_temp = self.make_parent(success=True)
        bad, bad_temp = self.make_parent(success=False)
        self.addCleanup(good_temp.cleanup)
        self.addCleanup(bad_temp.cleanup)
        candidates = [
            {"parent": good, "task": "rotate", "trajectory_id": "000001"},
            {"parent": bad, "task": "rotate", "trajectory_id": "000002"},
        ]
        records, exclusions = module.inspect_candidates(
            candidates,
            source_revision="revision",
            payload_root=good.parents[1],
        )
        self.assertEqual(1, len(records))
        self.assertEqual("000002", exclusions[0]["trajectory_id"])
        self.assertIn("success", exclusions[0]["reason"])


if __name__ == "__main__":
    unittest.main()
