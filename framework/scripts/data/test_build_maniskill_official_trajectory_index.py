#!/usr/bin/env python3
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import h5py
import numpy as np

MODULE_PATH = Path(__file__).with_name("build_maniskill_official_trajectory_index.py")
spec = importlib.util.spec_from_file_location("maniskill_index", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class ManiSkillOfficialIndexTest(unittest.TestCase):
    def make_archive(self, *, success: bool = True, bad_alignment: bool = False, mixed: bool = False) -> tuple[Path, dict, tempfile.TemporaryDirectory]:
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        task = "PegInsertionSide-v1"
        h5_path = root / "trajectory.none.pd_joint_pos.physx_cpu.h5"
        episode_successes = [success, False] if mixed else [success]
        with h5py.File(h5_path, "w") as handle:
            for episode_id, episode_success in enumerate(episode_successes):
                traj = handle.create_group(f"traj_{episode_id}")
                traj.create_dataset("actions", data=np.zeros((6, 8), dtype=np.float32))
                traj.create_dataset("terminated", data=np.array([0, 0, 0, 0, 0, episode_success], dtype=np.bool_))
                traj.create_dataset("truncated", data=np.zeros(6, dtype=np.bool_))
                traj.create_dataset("success", data=np.array([0, 0, 0, 0, 0, episode_success], dtype=np.bool_))
                states = traj.create_group("env_states")
                states.create_dataset("actors/peg", data=np.zeros((6 if bad_alignment else 7, 13), dtype=np.float32))
        metadata = {
            "env_info": {"env_id": task, "max_episode_steps": 200, "env_kwargs": {"obs_mode": "none", "control_mode": "pd_joint_pos", "sim_backend": "physx_cpu"}},
            "source_type": "motionplanning",
            "source_desc": "official synthetic fixture",
            "episodes": [
                {"episode_id": episode_id, "elapsed_steps": 6, "control_mode": "pd_joint_pos", "reset_kwargs": {"seed": 7 + episode_id}, "info": {"success": episode_success}, "success": episode_success}
                for episode_id, episode_success in enumerate(episode_successes)
            ],
        }
        json_path = h5_path.with_suffix(".json")
        json_path.write_text(json.dumps(metadata))
        archive = root / f"{task}.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.write(h5_path, f"demos/{task}/{h5_path.name}")
            bundle.write(json_path, f"demos/{task}/{json_path.name}")
        payload = archive.read_bytes()
        row = {"task": task, "path": f"demos/{task}.zip", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        return archive, row, temp

    def test_builds_successful_parent_with_native_alignment_and_complete_nonempty_phases(self) -> None:
        archive, row, temp = self.make_archive()
        self.addCleanup(temp.cleanup)

        records, receipt = module.inspect_archive(archive, row, source_revision="fixture-revision")

        self.assertEqual(1, len(records))
        record = records[0]
        self.assertEqual("maniskill", record["simulator"])
        self.assertEqual("insertion_assembly", record["canonical_task"]["family"])
        self.assertEqual(6, record["native_transition_count"])
        self.assertEqual("T_plus_1_env_states_to_T_actions", record["native_payload"]["temporal_alignment"])
        self.assertEqual("none", record["native_payload"]["obs_mode"])
        self.assertTrue(record["parent_success_evidence"]["success"])
        self.assertFalse(any("candidate_g" in json.dumps(record).lower() for _ in [0]))
        phases = record["phase_views"]
        self.assertTrue(all(p["start"] <= p["end"] and p["independent_success"] is False for p in phases))
        self.assertEqual(list(range(6)), [i for p in phases for i in range(p["start"], p["end"] + 1)])
        self.assertEqual(1, receipt["parents"])

    def test_rejects_episode_without_native_success_evidence(self) -> None:
        archive, row, temp = self.make_archive(success=False)
        self.addCleanup(temp.cleanup)
        with self.assertRaisesRegex(ValueError, "success evidence"):
            module.inspect_archive(archive, row, source_revision="fixture-revision")

    def test_mixed_official_archive_indexes_only_successful_parents_and_audits_exclusions(self) -> None:
        archive, row, temp = self.make_archive(mixed=True)
        self.addCleanup(temp.cleanup)

        records, receipt = module.inspect_archive(archive, row, source_revision="fixture-revision")

        self.assertEqual(1, len(records))
        self.assertEqual(1, receipt["excluded_without_success_evidence"])
        self.assertEqual(2, receipt["native_episode_candidates"])

    def test_rejects_T_plus_1_state_alignment_violation(self) -> None:
        archive, row, temp = self.make_archive(bad_alignment=True)
        self.addCleanup(temp.cleanup)
        with self.assertRaisesRegex(ValueError, "T\+1"):
            module.inspect_archive(archive, row, source_revision="fixture-revision")

    def test_parent_identity_includes_full_native_hdf5_member_not_just_basename(self) -> None:
        first = module.parent_episode_id("PushT-v1", "PushT-v1/motionplanning/trajectory.h5", "traj_0")
        second = module.parent_episode_id("PushT-v1", "PushT-v1/rl/trajectory.h5", "traj_0")
        self.assertNotEqual(first, second)

    def test_maps_selected_underrepresented_tasks(self) -> None:
        expected = {
            "LiftPegUpright-v1": "orientation_manipulation",
            "PegInsertionSide-v1": "insertion_assembly",
            "PlugCharger-v1": "insertion_assembly",
            "PokeCube-v1": "tool_or_contact_operation",
            "PullCubeTool-v1": "tool_or_contact_operation",
            "PushT-v1": "planar_push",
            "RollBall-v1": "rolling_contact",
            "DrawTriangle-v1": "tool_or_contact_operation",
            "StackPyramid-v1": "multi_object_arrangement",
        }
        self.assertEqual(expected, {task: module.infer_semantics(task)["family"] for task in expected})


if __name__ == "__main__":
    unittest.main()
