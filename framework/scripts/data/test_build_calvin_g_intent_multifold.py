#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).with_name("build_calvin_g_intent_multifold.py")
spec = importlib.util.spec_from_file_location("g_intent_multifold", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load {MODULE_PATH}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GIntentMultifoldTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scenes = ("calvin_scene_A", "calvin_scene_B", "calvin_scene_C")
        self.rows: list[dict[str, object]] = []
        annotation_index = 0
        for task_index, task in enumerate(module.ALL_INTENT_TASKS):
            for scene_index, scene in enumerate(self.scenes):
                base = scene_index * 100000 + task_index * 2000
                for item in range(20):
                    start = base + item * 10
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
        self.baseline = [
            row
            for task in module.ALL_INTENT_TASKS
            for scene in self.scenes
            for row in [
                next(
                    item
                    for item in self.rows
                    if item["task"] == task and item["scene"] == scene
                )
            ]
        ]

    def test_g_contract_is_intent_causal_and_treats_color_as_nuisance(self) -> None:
        contract = module.build_g_contract()
        self.assertEqual(
            {"interaction_region", "entity_motion", "goal_relation", "phase"},
            set(contract["causal_axes"]),
        )
        self.assertIn("visual_color", contract["nuisance_axes"])
        self.assertNotIn("visual_color", contract["causal_axes"])
        self.assertIn("robot_joint_ids", contract["adapter_only_axes"])
        self.assertEqual(
            "task_frame_low_dimensional_guidance",
            contract["representation_scope"],
        )

    def test_builds_balanced_connected_block_and_inverse_articulation_folds(self) -> None:
        folds = module.build_factor_folds()
        self.assertEqual(12, len(folds))
        self.assertEqual(12, len({fold["fold_id"] for fold in folds}))
        for fold in folds:
            source = set(fold["source_tasks"])
            heldout = set(fold["heldout_tasks"])
            self.assertTrue(source.isdisjoint(heldout))
            self.assertEqual(9, len(source))
            self.assertEqual(3, len(heldout))
            self.assertEqual(set(module.ALL_INTENT_TASKS), source | heldout)
            self.assertTrue(set(module.CONTAINER_TASKS) <= source)
            self.assertEqual(2, len(heldout & set(module.BLOCK_TASKS)))
            self.assertEqual(1, len(heldout & set(module.ARTICULATION_TASKS)))

        block_signatures = {
            (fold["block_fold_id"], tuple(sorted(set(fold["heldout_tasks"]) & set(module.BLOCK_TASKS))))
            for fold in folds
        }
        articulation_signatures = {
            (
                fold["articulation_fold_id"],
                tuple(sorted(set(fold["heldout_tasks"]) & set(module.ARTICULATION_TASKS))),
            )
            for fold in folds
        }
        self.assertEqual(3, len(block_signatures))
        self.assertEqual(4, len(articulation_signatures))
        self.assertEqual(
            {task: 4 for task in module.BLOCK_TASKS},
            dict(Counter(task for fold in folds for task in set(fold["heldout_tasks"]) & set(module.BLOCK_TASKS))),
        )
        self.assertEqual(
            {task: 3 for task in module.ARTICULATION_TASKS},
            dict(
                Counter(
                    task
                    for fold in folds
                    for task in set(fold["heldout_tasks"]) & set(module.ARTICULATION_TASKS)
                )
            ),
        )

    def test_superset_is_deterministic_balanced_nonoverlapping_and_retains_baseline(self) -> None:
        first = module.select_intent_superset(self.rows, self.baseline, 12, 20260814)
        second = module.select_intent_superset(list(reversed(self.rows)), self.baseline, 12, 20260814)
        self.assertEqual(
            [module.segment_key(row) for row in first],
            [module.segment_key(row) for row in second],
        )
        self.assertEqual(12 * 3 * 12, len(first))
        self.assertTrue(
            {module.segment_key(row) for row in self.baseline}
            <= {module.segment_key(row) for row in first}
        )
        for task in module.ALL_INTENT_TASKS:
            for scene in self.scenes:
                self.assertEqual(
                    12,
                    sum(row["task"] == task and row["scene"] == scene for row in first),
                )
        intervals = sorted((int(row["start"]), int(row["end"])) for row in first)
        self.assertTrue(all(left[1] < right[0] for left, right in zip(intervals, intervals[1:])))

    def test_each_fold_uses_full_source_cells_fixed_heldout_cells_and_no_leakage(self) -> None:
        superset = module.select_intent_superset(self.rows, self.baseline, 12, 20260814)
        for fold_spec in module.build_factor_folds():
            manifest = module.materialize_fold(
                superset,
                fold_spec,
                source_per_task_scene=12,
                heldout_per_task_scene=3,
                seed=20260814,
            )
            segments = manifest["segments_detail"]
            self.assertEqual(9 * 3 * 12 + 3 * 3 * 3, len(segments))
            source_tasks = set(manifest["source_tasks"])
            heldout_tasks = set(manifest["heldout_tasks"])
            self.assertTrue(source_tasks.isdisjoint(heldout_tasks))
            self.assertEqual(
                source_tasks,
                {row["task"] for row in segments if row["split"] == "source"},
            )
            self.assertEqual(
                heldout_tasks,
                {row["task"] for row in segments if row["split"] == "heldout"},
            )
            intervals = [(int(row["start"]), int(row["end"])) for row in segments]
            self.assertEqual(len(intervals), len(set(intervals)))
            for task in source_tasks:
                for scene in self.scenes:
                    self.assertEqual(
                        12,
                        sum(
                            row["task"] == task and row["scene"] == scene
                            for row in segments
                        ),
                    )
            for task in heldout_tasks:
                for scene in self.scenes:
                    self.assertEqual(
                        3,
                        sum(
                            row["task"] == task and row["scene"] == scene
                            for row in segments
                        ),
                    )
            self.assertEqual("audit_only", manifest["counterfactual_policy"]["usage"])
            self.assertFalse(manifest["counterfactual_policy"]["expert_action_label"])

    def test_heldout_episode_pool_is_fixed_per_task_across_combined_folds(self) -> None:
        superset = module.select_intent_superset(self.rows, self.baseline, 12, 20260814)
        heldout_keys_by_task: dict[str, set[tuple[tuple[str, str, int, int], ...]]] = {}
        for fold_spec in module.build_factor_folds():
            manifest = module.materialize_fold(superset, fold_spec, 12, 3, 20260814)
            for task in manifest["heldout_tasks"]:
                keys = tuple(
                    sorted(
                        module.segment_key(row)
                        for row in manifest["segments_detail"]
                        if row["split"] == "heldout" and row["task"] == task
                    )
                )
                heldout_keys_by_task.setdefault(task, set()).add(keys)
        self.assertEqual(
            {task: 1 for task in module.BLOCK_TASKS + module.ARTICULATION_TASKS},
            {task: len(pools) for task, pools in heldout_keys_by_task.items()},
        )


if __name__ == "__main__":
    unittest.main()
