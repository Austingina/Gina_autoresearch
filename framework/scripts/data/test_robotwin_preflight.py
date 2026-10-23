#!/usr/bin/env python3
"""Pure stdlib tests for the RoboTwin place_empty_cup preflight."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, REPO.as_posix())

from framework.sim_data.robotwin_preflight import (  # noqa: E402
    ASSET_REQUIREMENTS,
    REQUIRED_PACKAGES,
    PreflightInputs,
    check_preflight,
)

PINNED_COMMIT = "ea8b21121ebb3cd201ff5b3fe361944ac94eda3f"
EXPECTED_PACKAGES = (
    "torch",
    "torchvision",
    "numpy",
    "cv2",
    "transforms3d",
    "sapien",
    "scipy",
    "mplib",
    "gymnasium",
    "trimesh",
    "open3d",
    "imageio",
    "pydantic",
    "yaml",
    "zarr",
    "h5py",
)
EXPECTED_ASSETS = {
    "cup": "assets/objects/021_cup",
    "coaster": "assets/objects/019_coaster",
    "embodiment": "assets/embodiments/aloha-agilex",
    "background_texture": "assets/background_texture",
}


def _write(path: Path, text: str = "payload") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class RoboTwinPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name) / "RoboTwin"
        self.root.mkdir()
        for relative in EXPECTED_ASSETS.values():
            _write(self.root / relative / "sentinel.bin", "non-empty")
        _write(
            self.root / "envs/place_empty_cup.py",
            "class place_empty_cup:\n"
            "    def play_once(self):\n        return {}\n"
            "    def check_success(self):\n        return True\n",
        )
        _write(
            self.root / "scripts/collect_data.py",
            "def admit(task):\n"
            "    return (task.plan_success and task.check_success() "
            "and task.planned_joints_legal()[0])\n",
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def inputs(self, **overrides: object) -> PreflightInputs:
        values: dict[str, object] = {
            "project_root": self.root,
            "expected_root": self.root,
            "expected_commit": PINNED_COMMIT,
            "environment": {"PYTHONNOUSERSITE": "1"},
            "pilot_config": {"episode_num": 1, "max_attempts": 25},
            "commit_reader": lambda _root: PINNED_COMMIT,
            "package_finder": lambda _name: object(),
        }
        values.update(overrides)
        return PreflightInputs(**values)

    def test_declares_required_packages_and_four_asset_classes(self) -> None:
        self.assertEqual(REQUIRED_PACKAGES, EXPECTED_PACKAGES)
        self.assertEqual(ASSET_REQUIREMENTS, EXPECTED_ASSETS)

    def test_happy_path_passes_without_mutating_project(self) -> None:
        before = _tree_digest(self.root)
        report = check_preflight(self.inputs())
        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(_tree_digest(self.root), before)

    def test_project_root_and_commit_must_match_exactly(self) -> None:
        other = self.root.parent / "other"
        other.mkdir()
        root_report = check_preflight(self.inputs(expected_root=other))
        commit_report = check_preflight(
            self.inputs(commit_reader=lambda _root: "0" * 40)
        )
        self.assertFalse(root_report.by_name("project_root").ok)
        self.assertFalse(commit_report.by_name("git_commit").ok)

    def test_python_no_user_site_uses_cpython_nonempty_string_semantics(self) -> None:
        for accepted in ("1", "0", "false", "anything"):
            with self.subTest(accepted=accepted):
                report = check_preflight(
                    self.inputs(environment={"PYTHONNOUSERSITE": accepted})
                )
                self.assertTrue(report.by_name("python_no_user_site").ok)
        for rejected in (None, ""):
            with self.subTest(rejected=rejected):
                environment = {} if rejected is None else {"PYTHONNOUSERSITE": rejected}
                report = check_preflight(self.inputs(environment=environment))
                self.assertFalse(report.by_name("python_no_user_site").ok)

    def test_every_required_package_must_be_discoverable(self) -> None:
        missing = REQUIRED_PACKAGES[-1]
        report = check_preflight(
            self.inputs(package_finder=lambda name: None if name == missing else object())
        )
        check = report.by_name("required_packages")
        self.assertFalse(check.ok)
        self.assertEqual(check.details["missing"], [missing])

    def test_each_asset_class_must_contain_a_nonempty_file(self) -> None:
        for label, relative in ASSET_REQUIREMENTS.items():
            with self.subTest(asset=label):
                asset = self.root / relative
                for path in sorted(asset.rglob("*"), reverse=True):
                    if path.is_file():
                        path.unlink()
                report = check_preflight(self.inputs())
                self.assertFalse(report.by_name(f"asset:{label}").ok)
                _write(asset / "sentinel.bin", "restored")

    def test_task_source_requires_play_once_and_check_success_definitions(self) -> None:
        source = self.root / "envs/place_empty_cup.py"
        for missing, body in (
            ("play_once", "class place_empty_cup:\n    def check_success(self): return True\n"),
            ("check_success", "class place_empty_cup:\n    def play_once(self): return {}\n"),
        ):
            with self.subTest(missing=missing):
                source.write_text(body, encoding="utf-8")
                report = check_preflight(self.inputs())
                self.assertFalse(report.by_name("task_source_contract").ok)

    def test_collector_admission_requires_all_three_native_gates(self) -> None:
        collector = self.root / "scripts/collect_data.py"
        gates = ("plan_success", "check_success", "planned_joints_legal")
        for missing in gates:
            with self.subTest(missing=missing):
                used = [gate for gate in gates if gate != missing]
                expression = " and ".join(
                    "task.plan_success" if gate == "plan_success" else f"task.{gate}()"
                    for gate in used
                )
                collector.write_text(
                    f"def admit(task):\n    return {expression}\n", encoding="utf-8"
                )
                report = check_preflight(self.inputs())
                self.assertFalse(report.by_name("collector_native_gate").ok)

    def test_fixed_pilot_is_one_episode_and_has_positive_attempt_bound(self) -> None:
        bad_configs = (
            {},
            {"episode_num": 2, "max_attempts": 25},
            {"episode_num": 1},
            {"episode_num": 1, "max_attempts": 0},
            {"episode_num": True, "max_attempts": 25},
            {"episode_num": 1, "max_attempts": True},
        )
        for config in bad_configs:
            with self.subTest(config=config):
                report = check_preflight(self.inputs(pilot_config=config))
                self.assertFalse(report.by_name("bounded_pilot_config").ok)

    def test_background_texture_is_conditional_on_random_background(self) -> None:
        background = self.root / ASSET_REQUIREMENTS["background_texture"]
        for child in background.iterdir():
            child.unlink()
        background.rmdir()

        clean_config = {
            "episode_num": 1,
            "max_attempts": 25,
            "random_background": False,
        }
        clean_report = check_preflight(self.inputs(pilot_config=clean_config))
        self.assertTrue(clean_report.by_name("asset:background_texture").ok)
        self.assertFalse(
            clean_report.by_name("asset:background_texture").details["required"]
        )

        randomized_config = dict(clean_config, random_background=True)
        randomized_report = check_preflight(
            self.inputs(pilot_config=randomized_config)
        )
        self.assertFalse(randomized_report.by_name("asset:background_texture").ok)
        self.assertTrue(
            randomized_report.by_name("asset:background_texture").details["required"]
        )

    def test_missing_or_malformed_sources_fail_closed(self) -> None:
        (self.root / "envs/place_empty_cup.py").write_text("not python !!!", encoding="utf-8")
        (self.root / "scripts/collect_data.py").unlink()
        report = check_preflight(self.inputs())
        self.assertFalse(report.ok)
        self.assertFalse(report.by_name("task_source_contract").ok)
        self.assertFalse(report.by_name("collector_native_gate").ok)


if __name__ == "__main__":
    unittest.main(verbosity=2)
