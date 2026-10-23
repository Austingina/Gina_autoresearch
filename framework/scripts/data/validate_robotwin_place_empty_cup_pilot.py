#!/usr/bin/env python3
"""Independent full readback for one RoboTwin native pilot artifact."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_decoder(robotwin_root: Path):
    path = robotwin_root / "data/decode_image_bit.py"
    spec = importlib.util.spec_from_file_location("pinned_robotwin_image_codec", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load codec: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decode_image_bit


def validate(output_dir: Path, robotwin_root: Path) -> dict[str, Any]:
    receipt_path = output_dir / "acceptance_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    artifact = Path(receipt["artifact"]["path"])
    checks: dict[str, bool] = {}
    checks["artifact_under_output_dir"] = output_dir.resolve() in artifact.resolve().parents
    checks["artifact_exists_nonempty"] = artifact.is_file() and artifact.stat().st_size > 0
    actual_sha256 = _sha256(artifact)
    checks["artifact_sha256"] = actual_sha256 == receipt["artifact"]["sha256"]
    checks["native_oracle_success"] = receipt.get("native_oracle_success") is True
    checks["quality_passed"] = receipt.get("replay", {}).get("quality", {}).get("passed") is True

    decoder = _load_decoder(robotwin_root)
    camera_report: dict[str, Any] = {}
    with h5py.File(artifact, "r") as handle:
        checks["source_format"] = handle.attrs.get("source_format") == "RoboTwin"
        checks["source_path"] = handle.attrs.get("source_path") == "native_collection"
        version = handle["data_format_version"][()]
        if isinstance(version, bytes):
            version = version.decode("utf-8")
        checks["data_format_version"] = version == "v1.0"
        frequency = int(handle["additional_info/frequency"][()])
        checks["frequency_positive"] = frequency > 0

        state_lengths = {name: len(dataset) for name, dataset in handle["state"].items()}
        action_lengths = {name: len(dataset) for name, dataset in handle["action"].items()}
        all_lengths = list(state_lengths.values()) + list(action_lengths.values())
        checks["state_action_present"] = bool(state_lengths) and bool(action_lengths)
        checks["state_action_aligned"] = bool(all_lengths) and len(set(all_lengths)) == 1 and all_lengths[0] > 0
        transition_count = all_lengths[0] if all_lengths else 0

        expected_cameras = {"cam_head", "cam_left_wrist", "cam_right_wrist"}
        actual_cameras = set(handle["vision"].keys())
        checks["expected_cameras"] = expected_cameras <= actual_cameras
        for camera_name in sorted(actual_cameras):
            camera = handle[f"vision/{camera_name}"]
            colors = camera["colors"]
            decoded = np.asarray(decoder(colors[()]))
            shape = tuple(int(value) for value in camera["shape"][()])
            camera_report[camera_name] = {
                "encoded_frames": len(colors),
                "decoded_shape": list(decoded.shape),
                "declared_frame_shape": list(shape),
                "min": int(decoded.min()),
                "max": int(decoded.max()),
                "std": float(decoded.std()),
            }
            checks[f"{camera_name}:length"] = len(colors) == transition_count
            checks[f"{camera_name}:shape"] = decoded.ndim == 4 and tuple(decoded.shape[1:]) == shape
            checks[f"{camera_name}:uint8"] = decoded.dtype == np.uint8
            checks[f"{camera_name}:nonblank"] = int(decoded.max()) > int(decoded.min()) and float(decoded.std()) > 1.0

    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "output_dir": str(output_dir.resolve()),
        "artifact": str(artifact.resolve()),
        "artifact_sha256": actual_sha256,
        "checks": checks,
        "transition_count": transition_count,
        "state_lengths": state_lengths,
        "action_lengths": action_lengths,
        "cameras": camera_report,
        "training_manifest_eligible": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.output_dir, args.robotwin_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
