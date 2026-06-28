#!/usr/bin/env python3
"""Preflight checks for visual guidance field autoresearch."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "program.md",
    "prepare.py",
    "experiment.py",
    "run_experiment.py",
    "goals/schema.py",
    "goals/paths.py",
    "configs/default.yaml",
    "docs/EXPERIMENT_PLAN.md",
    "docs/REFERENCES.md",
    "docs/EXPERIMENT_PROTOCOL.md",
    "docs/DATA_ISOLATION.md",
    "data/README.md",
    "scripts/learn_guidance_field.py",
    "scripts/import_isaac_demos.py",
    "scripts/init_workspace.sh",
    "results.tsv",
]

OPTIONAL_PATHS = {
    "ISAACLAB_ROOT": "Isaac lift DAgger + wander eval",
    "HIEVT_ROOT": "HIEVT tracking baseline",
    "EVT_ROOT": "EVT anchor extraction",
    "DUALGF_ROOT": "DualGF target+support field training (local)",
    "TARGF_ROOT": "TarGF score-matching field (clone from GitHub if missing)",
}


def main() -> int:
    ok = True
    print("=== AnchorMask + GuidanceField Preflight ===\n")

    for name in REQUIRED:
        path = ROOT / name
        status = "OK" if path.is_file() else "MISSING"
        print(f"  [{status}] {name}")
        if status == "MISSING":
            ok = False

    print("\n--- External repos ---")
    defaults = {
        "ISAACLAB_ROOT": "/home/l/code/IsaacLab",
        "HIEVT_ROOT": "/home/l/code/Hierarchical-Instruction-aware-Embodied-Visual-Tracking-master",
        "EVT_ROOT": "/home/l/code/Offline_RL_Active_Tracking",
        "DUALGF_ROOT": "/home/l/code/DualGF",
        "TARGF_ROOT": "/home/l/code/TarGF",
    }
    for env_var, desc in OPTIONAL_PATHS.items():
        path = Path(os.environ.get(env_var, defaults.get(env_var, "")))
        exists = path.is_dir()
        print(f"  [{'OK' if exists else 'WARN'}] {env_var}={path} ({desc})")

    print("\n--- Import + schema ---")
    try:
        sys.path.insert(0, str(ROOT))
        from experiment import ExperimentConfig, build_policy_goal
        from goals.schema import GuidanceMode, VisualPolicyGoal
        g = build_policy_goal(ExperimentConfig(guidance_mode=GuidanceMode.ANCHOR_FIELD_RULE.value))
        assert g.guidance is not None
        print("  [OK] imports + build_policy_goal")
    except Exception as e:
        print(f"  [FAIL] {e}")
        ok = False

    print("\n=== " + ("READY" if ok else "NOT READY") + " ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
