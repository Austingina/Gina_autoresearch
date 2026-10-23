#!/usr/bin/env python3
"""Contract tests for manifest-exact CALVIN segment selection."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile


MODULE_PATH = Path(__file__).with_name("train_eval_state_chunk_flow_policy.py")
SPEC = importlib.util.spec_from_file_location("chunk_flow", MODULE_PATH)
assert SPEC and SPEC.loader
chunk_flow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chunk_flow)


def main() -> None:
    segments = [
        {"task": "push_red_block_left", "start": 10, "end": 19},
        {"task": "push_red_block_left", "start": 20, "end": 29},
        {"task": "push_blue_block_left", "start": 30, "end": 39},
    ]
    manifest = {
        "fold_id": "fold_test",
        "segments_detail": [
            {
                "task": "push_red_block_left",
                "start": 10,
                "end": 19,
                "split": "source",
            },
            {
                "task": "push_red_block_left",
                "start": 20,
                "end": 29,
                "split": "heldout",
            },
            {
                "task": "push_blue_block_left",
                "start": 30,
                "end": 39,
                "split": "heldout",
            },
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fold.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        selected, receipt = chunk_flow.select_segments_from_manifest(
            segments, path, "source"
        )
        assert selected == [segments[0]]
        assert receipt["fold_id"] == "fold_test"
        assert receipt["split"] == "source"
        assert receipt["selected_segments"] == 1
        assert len(receipt["manifest_sha256"]) == 64

        missing_manifest = dict(manifest)
        missing_manifest["segments_detail"] = [
            {
                "task": "push_pink_block_right",
                "start": 99,
                "end": 109,
                "split": "source",
            }
        ]
        path.write_text(json.dumps(missing_manifest), encoding="utf-8")
        try:
            chunk_flow.select_segments_from_manifest(segments, path, "source")
        except RuntimeError as exc:
            assert "missing from dataset annotations" in str(exc)
        else:
            raise AssertionError("missing manifest segment must fail closed")

        duplicate_manifest = dict(manifest)
        duplicate_manifest["segments_detail"] = [
            manifest["segments_detail"][0], manifest["segments_detail"][0]
        ]
        path.write_text(json.dumps(duplicate_manifest), encoding="utf-8")
        try:
            chunk_flow.select_segments_from_manifest(segments, path, "source")
        except RuntimeError as exc:
            assert "duplicate" in str(exc).lower()
        else:
            raise AssertionError("duplicate manifest segment must fail closed")

        primitive_path = Path(tmp) / "primitive_geometry.json"
        primitive_path.write_text(
            json.dumps(
                {
                    "push_into_drawer": {
                        "target_center": [0.1, 0.2, 0.3],
                        "flow_dir": [1.0, 0.0, 0.0],
                        "median_displacement": 0.1,
                    },
                    "place_in_slider": {
                        "target_center": [0.4, 0.5, 0.6],
                        "flow_dir": [0.0, 1.0, 0.0],
                        "median_displacement": 0.2,
                    },
                }
            ),
            encoding="utf-8",
        )
        primitive, primitive_receipt = chunk_flow.load_fixed_primitive_geometry(
            primitive_path, {"push_into_drawer", "place_in_slider"}
        )
        assert set(primitive) == {"push_into_drawer", "place_in_slider"}
        assert len(primitive_receipt["sha256"]) == 64

    print("manifest-exact segment selection: OK")


if __name__ == "__main__":
    main()
