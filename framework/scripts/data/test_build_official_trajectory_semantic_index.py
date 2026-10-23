#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


MODULE_PATH = Path(__file__).with_name("build_official_trajectory_semantic_index.py")
spec = importlib.util.spec_from_file_location("official_semantic_index", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load {MODULE_PATH}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class OfficialTrajectorySemanticIndexTest(unittest.TestCase):
    def test_partition_phases_is_contiguous_complete_and_nonempty(self) -> None:
        phases = module.partition_progress_phases(
            [0.0, 0.0, 0.01, 0.03, 0.07, 0.09, 0.1, 0.1],
            motion_floor=0.002,
        )
        self.assertEqual(
            ["approach", "contact_transition_proxy", "sustained_interaction", "completion"],
            [row["phase"] for row in phases],
        )
        self.assertEqual(0, phases[0]["start"])
        self.assertEqual(7, phases[-1]["end"])
        self.assertTrue(all(row["start"] <= row["end"] for row in phases))
        self.assertTrue(
            all(left["end"] + 1 == right["start"] for left, right in zip(phases, phases[1:]))
        )
        self.assertEqual("state_motion_proxy_v1", phases[1]["boundary_evidence"])

    def test_calvin_block_progress_uses_task_object_and_direction(self) -> None:
        scene = np.zeros((4, 24), dtype=np.float32)
        scene[:, 12] = [0.20, 0.19, 0.16, 0.10]
        scene[:, 6] = [0.0, 0.4, 0.8, 1.2]  # unrelated red block motion
        progress = module.calvin_task_progress(scene, "push_blue_block_left", "calvin_scene_B")
        np.testing.assert_allclose(progress, [0.0, 0.01, 0.04, 0.10], atol=1e-6)

    def test_calvin_block_progress_uses_scene_specific_object_order(self) -> None:
        scene_a = np.zeros((4, 24), dtype=np.float32)
        scene_a[:, 18] = [0.20, 0.19, 0.16, 0.10]  # red is slot 2 in scene A
        scene_c = np.zeros((4, 24), dtype=np.float32)
        scene_c[:, 12] = [0.20, 0.19, 0.16, 0.10]  # red is slot 1 in scene C
        np.testing.assert_allclose(
            module.calvin_task_progress(scene_a, "push_red_block_left", "calvin_scene_A"),
            [0.0, 0.01, 0.04, 0.10],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            module.calvin_task_progress(scene_c, "push_red_block_left", "calvin_scene_C"),
            [0.0, 0.01, 0.04, 0.10],
            atol=1e-6,
        )

    def test_build_calvin_index_preserves_parent_success_and_marks_phase_views_derived(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training = root / "training"
            training.mkdir()
            for frame_id, x in enumerate([0.20, 0.19, 0.16, 0.10], start=10):
                scene = np.zeros(24, dtype=np.float32)
                scene[12] = x
                np.savez_compressed(
                    training / f"episode_{frame_id:07d}.npz",
                    scene_obs=scene,
                    robot_obs=np.zeros(15, dtype=np.float32),
                    rel_actions=np.zeros(7, dtype=np.float32),
                )
            manifest = {
                "format": "fixture",
                "complete": True,
                "source_url": "https://official.example/calvin.zip",
                "segments_detail": [
                    {
                        "task": "push_blue_block_left",
                        "scene": "calvin_scene_A",
                        "start": 10,
                        "end": 13,
                        "annotation_index": 7,
                        "intent_family": "block_push",
                        "primitive": "push",
                        "entity": "blue_block",
                        "direction": "x-",
                    }
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            records = module.build_calvin_index(root, manifest_path)

        self.assertEqual(1, len(records))
        record = records[0]
        self.assertEqual("official_success_annotation", record["parent_success_evidence"]["kind"])
        self.assertTrue(record["parent_success_evidence"]["success"])
        self.assertEqual(4, len(record["phase_views"]))
        self.assertTrue(all(not phase["independent_success"] for phase in record["phase_views"]))
        self.assertNotIn("split", record["canonical_task"])

        def keys(value: object) -> set[str]:
            if isinstance(value, dict):
                return {str(key).lower() for key in value} | set().union(
                    *(keys(child) for child in value.values())
                )
            if isinstance(value, list):
                return set().union(*(keys(child) for child in value)) if value else set()
            return set()

        self.assertTrue({"g", "candidate_g", "provisional_g"}.isdisjoint(keys(record)))

    def test_partition_libero_actions_uses_gripper_transition_and_terminal_reward(self) -> None:
        actions = np.zeros((10, 7), dtype=np.float32)
        actions[:4, -1] = 1.0
        actions[4:, -1] = -1.0
        rewards = np.zeros(10, dtype=np.uint8)
        rewards[9] = 1
        phases = module.partition_libero_action_phases(actions, rewards)
        self.assertEqual(4, phases[1]["start"])
        self.assertEqual("gripper_transition_proxy_v1", phases[1]["boundary_evidence"])
        self.assertEqual(9, phases[-1]["start"])
        self.assertEqual(9, phases[-1]["end"])

    def test_build_libero_index_requires_terminal_success_and_preserves_native_group(self) -> None:
        try:
            import h5py
        except ImportError:
            self.skipTest("h5py is unavailable in this interpreter")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = root / "libero_object"
            suite.mkdir()
            payload = suite / "pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5"
            with h5py.File(payload, "w") as handle:
                data = handle.create_group("data")
                data.attrs["problem_info"] = json.dumps(
                    {"language_instruction": "pick up the milk and place it in the basket"}
                )
                data.attrs["env_args"] = json.dumps({"env_kwargs": {"control_freq": 20}})
                demo = data.create_group("demo_0")
                demo.attrs["num_samples"] = 6
                actions = np.zeros((6, 7), dtype=np.float32)
                actions[:2, -1] = 1
                actions[2:, -1] = -1
                demo.create_dataset("actions", data=actions)
                demo.create_dataset("states", data=np.zeros((6, 3), dtype=np.float32))
                demo.create_dataset("robot_states", data=np.zeros((6, 2), dtype=np.float32))
                demo.create_dataset("rewards", data=[0, 0, 0, 0, 0, 1])
                demo.create_dataset("dones", data=[0, 0, 0, 0, 0, 1])
                obs = demo.create_group("obs")
                obs.create_dataset("agentview_rgb", data=np.zeros((6, 2, 2, 3), dtype=np.uint8))
            file_hash = module._sha256(payload)
            manifest = {
                "source": "huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets",
                "source_revision": "fixture-revision",
                "suites": [
                    {
                        "suite": "libero_object",
                        "records": [
                            {
                                "file": payload.name,
                                "sha256": file_hash,
                                "demos": 1,
                                "transitions": 6,
                            }
                        ],
                    }
                ],
            }
            manifest_path = root / "inventory.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            records = module.build_libero_index(root, manifest_path, verify_hashes=True)

        self.assertEqual(1, len(records))
        self.assertEqual("data/demo_0", records[0]["native_payload"]["group"])
        self.assertEqual("pick_place", records[0]["canonical_task"]["family"])
        self.assertEqual(20, records[0]["native_payload"]["control_rate_hz"])
        self.assertTrue(records[0]["parent_success_evidence"]["success"])
        self.assertTrue(all(not view["independent_success"] for view in records[0]["phase_views"]))

    def test_summary_separates_parent_episodes_from_derived_phase_views(self) -> None:
        records = [
            {
                "simulator": "calvin",
                "native_task_id": "push_blue_block_left",
                "scene_or_suite": "calvin_scene_A",
                "native_frame_count": 8,
                "canonical_task": {"family": "block_push"},
                "phase_views": [{}, {}, {}, {}],
            },
            {
                "simulator": "libero",
                "native_task_id": "pick_up_milk",
                "scene_or_suite": "libero_object",
                "native_transition_count": 6,
                "canonical_task": {"family": "pick_place"},
                "phase_views": [{}, {}, {}, {}],
            },
        ]
        summary = module.summarize_records(records)
        self.assertEqual(2, summary["parent_trajectories"])
        self.assertEqual(14, summary["native_frames_or_transitions"])
        self.assertEqual(8, summary["derived_phase_views"])
        self.assertEqual({"calvin": 1, "libero": 1}, summary["by_simulator"])

    def test_write_outputs_roundtrips_jsonl_and_summary(self) -> None:
        records = [
            {
                "simulator": "calvin",
                "native_task_id": "open_drawer",
                "scene_or_suite": "calvin_scene_A",
                "native_frame_count": 4,
                "canonical_task": {"family": "articulation"},
                "phase_views": [{}, {}, {}, {}],
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            paths = module.write_outputs(records, output, unavailable_simulators={"robotwin": "no payload"})
            rows = [json.loads(line) for line in paths["records"].read_text().splitlines()]
            summary = json.loads(paths["summary"].read_text())
        self.assertEqual(records, rows)
        self.assertEqual(1, summary["parent_trajectories"])
        self.assertEqual("no payload", summary["unavailable_simulators"]["robotwin"])


if __name__ == "__main__":
    unittest.main()
