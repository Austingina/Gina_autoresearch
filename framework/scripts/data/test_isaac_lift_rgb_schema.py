#!/usr/bin/env python3
"""Static source contract for Gina Isaac lift RGB+Depth+Mask persistence."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
VISION = ROOT / "simulators/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/mdp/teacher_student/vision_observations.py"
CFG = ROOT / "simulators/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/vision_teacher_student_env_cfg.py"
MDP_INIT = ROOT / "simulators/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/mdp/__init__.py"
RECORDER = ROOT / "simulators/isaaclab/source/isaaclab/isaaclab/envs/mdp/recorders/recorders_cfg.py"
COLLECTOR = ROOT / "simulators/isaaclab/scripts/tools/lenient_teacher_datacollection_maskdepth.py"


def tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def class_node(module: ast.AST, name: str) -> ast.ClassDef:
    matches = [node for node in ast.walk(module) if isinstance(node, ast.ClassDef) and node.name == name]
    assert len(matches) == 1, (name, len(matches))
    return matches[0]


class IsaacLiftRGBSchemaContract(unittest.TestCase):
    def test_camera_annotators_include_rgb_depth_and_segmentation(self):
        text = CFG.read_text(encoding="utf-8")
        for token in ('"rgb"', '"instance_id_segmentation_fast"', '"distance_to_image_plane"'):
            self.assertIn(token, text)

    def test_camera_rgb_function_reads_rgb_and_returns_three_channels(self):
        module = tree(VISION)
        functions = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "camera_rgb"]
        self.assertEqual(len(functions), 1)
        source = ast.get_source_segment(VISION.read_text(encoding="utf-8"), functions[0]) or ""
        self.assertIn('"rgb"', source)
        self.assertIn("[..., :3]", source)
        self.assertIn("torch.float32", source)
        self.assertIn("255.0", source)

    def test_policy_observations_expose_both_rgb_views(self):
        observations = class_node(tree(CFG), "VisionObservationsCfg")
        policies = [
            node for node in observations.body if isinstance(node, ast.ClassDef) and node.name == "PolicyCfg"
        ]
        self.assertEqual(len(policies), 1)
        policy = policies[0]
        assigned = {
            target.id
            for node in policy.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertTrue({"ee_rgb", "third_person_rgb", "ee_depth", "third_person_depth", "ee_mask", "third_person_mask"} <= assigned)

    def test_camera_rgb_is_exported_through_lift_mdp(self):
        self.assertIn("camera_rgb", MDP_INIT.read_text(encoding="utf-8"))

    def test_generic_recorder_still_records_policy_observations(self):
        manager = class_node(tree(RECORDER), "ActionStateRecorderManagerCfg")
        assigned = {
            target.id
            for node in manager.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertIn("record_pre_step_flat_policy_observations", assigned)

    def test_collector_declares_rgb_depth_mask_schema(self):
        text = COLLECTOR.read_text(encoding="utf-8")
        self.assertIn("ee_rgb", text)
        self.assertIn("third_person_rgb", text)
        self.assertIn("ee_depth", text)
        self.assertIn("third_person_mask", text)

    def test_collector_fails_closed_on_all_zero_camera_outputs(self):
        text = COLLECTOR.read_text(encoding="utf-8")
        self.assertIn("def validate_nonzero_camera_outputs", text)
        self.assertIn("validate_nonzero_camera_outputs(env)", text)
        self.assertIn("torch.count_nonzero", text)
        self.assertIn("Vision preflight failed", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
