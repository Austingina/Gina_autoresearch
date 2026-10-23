#!/usr/bin/env python3
"""Fail-closed readback audit for a Gina Isaac lift RGB/depth/mask HDF5 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

EXPECTED_OBS = (
    "ee_rgb",
    "ee_depth",
    "ee_mask",
    "third_person_rgb",
    "third_person_depth",
    "third_person_mask",
    "student_state",
)
VISUAL_OBS = EXPECTED_OBS[:-1]


def scalar(value):
    return value.item() if hasattr(value, "item") else value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    path = args.hdf5.resolve()
    receipt = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "episode_checks": [],
    }
    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise RuntimeError("missing /data group")
        data = handle["data"]
        receipt["data_attrs"] = {key: scalar(value) for key, value in data.attrs.items()}
        for demo_name in sorted(data.keys()):
            demo = data[demo_name]
            if "obs" not in demo:
                raise RuntimeError(f"{demo_name}: missing obs")
            obs = demo["obs"]
            missing = [key for key in EXPECTED_OBS if key not in obs]
            if missing:
                raise RuntimeError(f"{demo_name}: missing obs keys {missing}")
            shapes = {key: list(obs[key].shape) for key in EXPECTED_OBS}
            lengths = {f"obs/{key}": int(obs[key].shape[0]) for key in EXPECTED_OBS}
            for key in ("actions", "env_actions"):
                if key not in demo:
                    raise RuntimeError(f"{demo_name}: missing {key}")
                lengths[key] = int(demo[key].shape[0])
            visual = {}
            for key in VISUAL_OBS:
                values = obs[key][...]
                visual[key] = {
                    "dtype": str(values.dtype),
                    "finite": bool(np.isfinite(values).all()),
                    "nonzero_fraction": float(np.count_nonzero(values) / values.size),
                    "min": float(np.nanmin(values)),
                    "max": float(np.nanmax(values)),
                }
            check = {
                "name": demo_name,
                "attrs": {key: scalar(value) for key, value in demo.attrs.items()},
                "shapes": shapes,
                "lengths": lengths,
                "lengths_consistent": len(set(lengths.values())) == 1,
                "visual": visual,
            }
            receipt["episode_checks"].append(check)
    receipt["status"] = (
        "PASS"
        if receipt["episode_checks"]
        and all(
            check["lengths_consistent"]
            and all(stats["finite"] and stats["nonzero_fraction"] > 0 for stats in check["visual"].values())
            for check in receipt["episode_checks"]
        )
        else "FAIL"
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
