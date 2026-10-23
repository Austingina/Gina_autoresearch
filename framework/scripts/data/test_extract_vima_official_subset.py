#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

MODULE_PATH = Path(__file__).with_name("extract_vima_official_subset.py")
spec = importlib.util.spec_from_file_location("extract_vima_official_subset", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class VimaOfficialSubsetTests(unittest.TestCase):
    def test_uniform_sample_spans_full_sorted_native_id_range(self) -> None:
        ids = [f"{value:06d}" for value in range(0, 100, 2)]
        selected = module.uniform_sample_ids(ids, 5)
        self.assertEqual(["000010", "000030", "000050", "000070", "000090"], selected)

    def test_inventory_requires_all_native_payload_components(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.zip"
            names = [
                "vima_v6/rotate/000001/action.pkl",
                "vima_v6/rotate/000001/obs.pkl",
                "vima_v6/rotate/000001/trajectory.pkl",
                "vima_v6/rotate/000001/rgb_front/0.jpg",
                "vima_v6/rotate/000001/rgb_front/1.jpg",
                "vima_v6/rotate/000001/rgb_top/0.jpg",
                "vima_v6/rotate/000001/rgb_top/1.jpg",
            ]
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                for name in names:
                    archive.writestr(name, b"payload")
            with zipfile.ZipFile(path) as archive:
                inventory = module.inventory_selected_tasks(archive, ("rotate",))
            self.assertEqual(["000001"], sorted(inventory["rotate"]))
            module.validate_native_member_set(inventory["rotate"]["000001"])

    def test_rejects_trajectory_missing_top_rgb(self) -> None:
        names = (
            "vima_v6/rotate/000001/action.pkl",
            "vima_v6/rotate/000001/obs.pkl",
            "vima_v6/rotate/000001/trajectory.pkl",
            "vima_v6/rotate/000001/rgb_front/0.jpg",
        )
        with self.assertRaisesRegex(ValueError, "rgb_top"):
            module.validate_native_member_set(names)


if __name__ == "__main__":
    unittest.main()
