#!/usr/bin/env python3
"""Static, no-rollout LIBERO middle-drawer runtime preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from libero.libero.envs.bddl_utils import robosuite_parse_problem

PINNED_HEAD = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
TASK = "open_the_middle_drawer_of_the_cabinet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.runtime_root.resolve()
    source = root / "source"
    package = source / "libero" / "libero"
    bddl = package / "bddl_files" / "libero_goal" / f"{TASK}.bddl"
    init_states = package / "init_files" / "libero_goal" / f"{TASK}.pruned_init"
    xml = package / "assets" / "articulated_objects" / "wooden_cabinet.xml"

    failures: list[str] = []
    head = git(source, "rev-parse", "HEAD")
    status = git(source, "status", "--short")
    if head != PINNED_HEAD:
        failures.append(f"HEAD mismatch: {head}")
    if status:
        failures.append("upstream source is dirty")

    for path in (bddl, init_states, xml):
        if not path.is_file() or path.stat().st_size <= 0:
            failures.append(f"missing or empty: {path}")

    parsed = robosuite_parse_problem(str(bddl))
    expected_goal = ["open", "wooden_cabinet_1_middle_region"]
    if expected_goal not in parsed.get("goal_state", []):
        failures.append(f"unexpected goal: {parsed.get('goal_state')!r}")
    if "wooden_cabinet_1_middle_region" not in parsed.get("obj_of_interest", []):
        failures.append(f"unexpected obj_of_interest: {parsed.get('obj_of_interest')!r}")

    states = torch.load(init_states, map_location="cpu", weights_only=False)
    state_array = np.asarray(states)
    if state_array.shape != (50, 79):
        failures.append(f"unexpected init-state shape: {state_array.shape}")
    if not np.isfinite(state_array).all():
        failures.append("init states contain non-finite values")

    tree = ET.parse(xml)
    xml_root = tree.getroot()
    joints = xml_root.findall(".//joint[@name='middle_level']")
    bodies = xml_root.findall(".//body[@name='cabinet_middle']")
    sites = xml_root.findall(".//site[@name='middle_region']")
    if len(joints) != 1:
        failures.append(f"middle_level joint count: {len(joints)}")
        joint_contract = None
    else:
        joint = joints[0]
        joint_contract = {
            "type": joint.get("type"),
            "axis": joint.get("axis"),
            "range": joint.get("range"),
            "limited": joint.get("limited"),
        }
        expected_joint = {"type": "slide", "axis": "0 1 0", "range": "-0.16 0.01", "limited": "true"}
        if joint_contract != expected_joint:
            failures.append(f"middle_level contract mismatch: {joint_contract!r}")
    if len(bodies) != 1:
        failures.append(f"cabinet_middle body count: {len(bodies)}")
    if len(sites) != 1:
        failures.append(f"middle_region site count: {len(sites)}")

    task_asset_xmls = {
        "wooden_cabinet": xml,
        "flat_stove": package / "assets" / "articulated_objects" / "flat_stove.xml",
        "wine_rack": package / "assets" / "turbosquid_objects" / "wine_rack" / "wine_rack.xml",
        "akita_black_bowl": package / "assets" / "stable_scanned_objects" / "akita_black_bowl" / "akita_black_bowl.xml",
        "cream_cheese": package / "assets" / "stable_hope_objects" / "cream_cheese" / "cream_cheese.xml",
        "wine_bottle": package / "assets" / "turbosquid_objects" / "wine_bottle" / "wine_bottle.xml",
        "plate": package / "assets" / "stable_scanned_objects" / "plate" / "plate.xml",
    }
    task_asset_receipts: dict[str, dict[str, object]] = {}
    for asset_name, asset_xml in task_asset_xmls.items():
        references: list[dict[str, object]] = []
        if not asset_xml.is_file() or asset_xml.stat().st_size <= 0:
            failures.append(f"missing task asset XML: {asset_name}: {asset_xml}")
            task_asset_receipts[asset_name] = {"path": str(asset_xml), "exists": False, "asset_references": references}
            continue
        asset_tree = ET.parse(asset_xml)
        for element in asset_tree.getroot().findall(".//*[@file]"):
            reference = element.get("file")
            assert reference is not None
            target = (asset_xml.parent / reference).resolve()
            exists = target.is_file() and target.stat().st_size > 0
            references.append({"reference": reference, "path": str(target), "exists": exists})
            if not exists:
                failures.append(f"missing {asset_name} XML asset: {reference}")
        task_asset_receipts[asset_name] = {
            "path": str(asset_xml),
            "exists": True,
            "sha256": sha256(asset_xml),
            "asset_references": references,
        }

    receipt = {
        "status": "PASS" if not failures else "FAIL",
        "scope": "static_no_rollout",
        "runtime_root": str(root),
        "libero_config_path": os.environ.get("LIBERO_CONFIG_PATH"),
        "source": {"head": head, "status_short": status, "pinned_head": PINNED_HEAD},
        "task": TASK,
        "bddl": {
            "path": str(bddl),
            "sha256": sha256(bddl),
            "language_instruction": parsed.get("language_instruction"),
            "goal_state": parsed.get("goal_state"),
            "obj_of_interest": parsed.get("obj_of_interest"),
        },
        "init_states": {
            "path": str(init_states),
            "sha256": sha256(init_states),
            "shape": list(state_array.shape),
            "dtype": str(state_array.dtype),
            "finite": bool(np.isfinite(state_array).all()),
        },
        "cabinet_xml": {
            "path": str(xml),
            "sha256": sha256(xml),
            "middle_level": joint_contract,
            "cabinet_middle_body_count": len(bodies),
            "middle_region_site_count": len(sites),
        },
        "task_assets": task_asset_receipts,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
