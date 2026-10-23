#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("extract_calvin_intent_factorial_subset.py")
spec = importlib.util.spec_from_file_location("intent_subset", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load {MODULE_PATH}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class IntentFactorialSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scenes = {
            "calvin_scene_A": [0, 99999],
            "calvin_scene_B": [100000, 199999],
            "calvin_scene_C": [200000, 299999],
        }
        self.rows = []
        annotation_index = 0
        for task_index, task in enumerate(module.SOURCE_TASKS + module.HELDOUT_TASKS):
            for scene_index, scene in enumerate(sorted(self.scenes)):
                base = scene_index * 100000 + task_index * 5000
                for item in range(20):
                    start = base + item * 11
                    self.rows.append(
                        {
                            "task": task,
                            "scene": scene,
                            "start": start,
                            "end": start + 4,
                            "annotation_index": annotation_index,
                        }
                    )
                    annotation_index += 1

    def test_preserves_baseline_and_balances_every_task_scene_cell(self) -> None:
        baseline = []
        for task in module.SOURCE_TASKS[:4]:
            for scene in sorted(self.scenes):
                rows = [r for r in self.rows if r["task"] == task and r["scene"] == scene]
                baseline.extend(rows[:12])
        for task in module.HELDOUT_TASKS[:2]:
            for scene in sorted(self.scenes):
                rows = [r for r in self.rows if r["task"] == task and r["scene"] == scene]
                baseline.extend(rows[:3])

        selected = module.select_intent_factorial_segments(
            self.rows,
            baseline,
            source_per_task_scene=12,
            heldout_per_task_scene=3,
            validation_modulus=5,
            seed=20260814,
        )
        selected_keys = {module.segment_key(row) for row in selected}
        self.assertTrue({module.segment_key(row) for row in baseline} <= selected_keys)
        self.assertEqual(8 * 3 * 12 + 4 * 3 * 3, len(selected))
        for task in module.SOURCE_TASKS:
            for scene in sorted(self.scenes):
                cell = [r for r in selected if r["task"] == task and r["scene"] == scene]
                self.assertEqual(12, len(cell))
                self.assertTrue(any(int(r["start"]) % 5 == 0 for r in cell))
        for task in module.HELDOUT_TASKS:
            for scene in sorted(self.scenes):
                cell = [r for r in selected if r["task"] == task and r["scene"] == scene]
                self.assertEqual(3, len(cell))

    def test_selection_is_deterministic_and_globally_nonoverlapping(self) -> None:
        first = module.select_intent_factorial_segments(
            self.rows, [], 12, 3, 5, 20260814
        )
        second = module.select_intent_factorial_segments(
            list(reversed(self.rows)), [], 12, 3, 5, 20260814
        )
        self.assertEqual(
            [module.segment_key(row) for row in first],
            [module.segment_key(row) for row in second],
        )
        intervals = sorted((int(r["start"]), int(r["end"])) for r in first)
        self.assertTrue(all(left[1] < right[0] for left, right in zip(intervals, intervals[1:])))

    def test_rejects_leakage_in_baseline(self) -> None:
        bad = dict(self.rows[0])
        bad["task"] = module.HELDOUT_TASKS[0]
        bad["split"] = "source"
        with self.assertRaisesRegex(ValueError, "baseline split mismatch"):
            module.select_intent_factorial_segments(
                self.rows, [bad], 12, 3, 5, 20260814
            )

    def test_schema_excludes_tasks_without_validated_translational_g(self) -> None:
        self.assertEqual(
            {"block_push", "articulation", "container_transport"},
            {item["family"] for item in module.INTENT_SCHEMA.values()},
        )
        self.assertNotIn("turn_on_led", module.INTENT_SCHEMA)
        self.assertNotIn("rotate_red_block_left", module.INTENT_SCHEMA)
        self.assertTrue(set(module.SOURCE_TASKS).isdisjoint(module.HELDOUT_TASKS))

    def test_loads_cached_full_annotation_and_metadata_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            annotation = Path(directory) / "full.npy"
            annotation.write_bytes(b"annotation")
            (root / "training/.hydra").mkdir(parents=True)
            (root / "training/.hydra/merged_config.yaml").write_bytes(b"config")
            (root / "training/scene_info.npy").write_bytes(b"scenes")
            metadata = module.load_local_metadata(annotation, root)
            self.assertEqual(
                {
                    "task_ABC_D/training/lang_annotations/auto_lang_ann.npy": b"annotation",
                    "task_ABC_D/training/.hydra/merged_config.yaml": b"config",
                    "task_ABC_D/training/scene_info.npy": b"scenes",
                },
                metadata,
            )

    def test_counterfactuals_are_audit_only_and_never_fabricate_actions(self) -> None:
        plan = module.build_counterfactual_audit_plan()
        self.assertEqual(
            {"zero_g", "sign_flip_f", "inverse_intent", "wrong_entity"},
            {row["condition"] for row in plan},
        )
        self.assertTrue(all(row["usage"] == "audit_only" for row in plan))
        self.assertTrue(all(row["expert_action_label"] is False for row in plan))
        self.assertTrue(any(row.get("target_task") == "close_drawer" for row in plan))


if __name__ == "__main__":
    unittest.main()
