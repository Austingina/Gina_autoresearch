#!/usr/bin/env python3
"""Run one bounded RoboTwin ``place_empty_cup`` native-expert pilot.

This Gina-owned launcher leaves the pinned RoboTwin source untouched.  It uses
fresh simulator instances per planning seed, requires the upstream native
admission conjunction, replays the saved joint path in a fresh instance, and
writes HDF5 only after replay Oracle success.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator, Mapping

from framework.scripts.data.collect_robotwin_bounded_pilot import (
    PilotQuotaError,
    replay_accepted_plan,
    search_accepted_plan,
)

TASK_NAME = "place_empty_cup"
XY_MAX_M = 0.035
Z_MAX_M = 0.015
TILT_MAX_DEG = 10.0
LINEAR_SPEED_MAX_MPS = 0.05
ANGULAR_SPEED_MAX_RADPS = 0.5


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", os.fspath(root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        timeout=10,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _robotwin_import_context(root: Path) -> Iterator[None]:
    previous_cwd = Path.cwd()
    root_text = os.fspath(root)
    inserted = root_text not in sys.path
    if inserted:
        sys.path.insert(0, root_text)
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        if inserted:
            sys.path.remove(root_text)


def _load_base_args(robotwin_root: Path, output_dir: Path) -> dict[str, Any]:
    import yaml

    config_path = robotwin_root / "env_cfg/task_config/demo_clean.yml"
    args = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    embodiment_root = robotwin_root / "assets/embodiments/aloha-agilex"
    embodiment_config = yaml.safe_load(
        (embodiment_root / "config.yml").read_text(encoding="utf-8")
    )
    args.update(
        {
            "task_name": TASK_NAME,
            "task_config": "gina_place_empty_cup_pilot_v1",
            "episode_num": 1,
            "use_seed": False,
            "save_path": os.fspath(output_dir),
            "left_robot_file": os.fspath(embodiment_root),
            "right_robot_file": os.fspath(embodiment_root),
            "left_embodiment_config": embodiment_config,
            "right_embodiment_config": embodiment_config,
            "dual_arm_embodied": True,
            "embodiment_name": "aloha-agilex",
            "need_plan": True,
            "save_data": False,
            "render_freq": 0,
        }
    )
    return args


def _rigid_speed(actor_wrapper: Any) -> tuple[float | None, float | None]:
    for component in actor_wrapper.actor.get_components():
        linear = getattr(component, "linear_velocity", None)
        angular = getattr(component, "angular_velocity", None)
        if linear is None or angular is None:
            continue
        try:
            import numpy as np

            return float(np.linalg.norm(linear)), float(np.linalg.norm(angular))
        except (TypeError, ValueError):
            continue
    return None, None


def _capture_upright_reference(task: Any) -> None:
    import numpy as np

    rotation = np.asarray(
        task.cup.get_pose().to_transformation_matrix()[:3, :3], dtype=float
    )
    world_up = np.array([0.0, 0.0, 1.0])
    alignments = rotation.T @ world_up
    local_axis_index = int(np.argmax(np.abs(alignments)))
    axis_sign = 1 if alignments[local_axis_index] >= 0.0 else -1
    upright_axis = axis_sign * rotation[:, local_axis_index]
    cosine = float(np.clip(np.dot(upright_axis, world_up), -1.0, 1.0))
    task._gina_upright_reference = {
        "local_axis_index": local_axis_index,
        "axis_sign": axis_sign,
        "initial_tilt_deg": float(math.degrees(math.acos(cosine))),
    }


def _quality_probe(task: Any) -> Mapping[str, Any]:
    import numpy as np

    cup_point = np.asarray(task.cup.get_functional_point(0, "pose").p, dtype=float)
    coaster_point = np.asarray(task.coaster.get_functional_point(0, "pose").p, dtype=float)
    delta = cup_point - coaster_point
    xy_error = float(np.linalg.norm(delta[:2]))
    z_error = float(abs(delta[2]))

    reference = getattr(task, "_gina_upright_reference", None)
    if not isinstance(reference, Mapping):
        raise RuntimeError("missing setup-time upright reference for quality probe")
    local_axis_index = int(reference["local_axis_index"])
    axis_sign = int(reference["axis_sign"])
    rotation = np.asarray(task.cup.get_pose().to_transformation_matrix()[:3, :3], dtype=float)
    upright_axis = axis_sign * rotation[:, local_axis_index]
    cosine = float(np.clip(np.dot(upright_axis, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))
    tilt_deg = float(math.degrees(math.acos(cosine)))
    linear_speed, angular_speed = _rigid_speed(task.cup)
    contact = bool(task.check_actors_contact(task.cup.get_name(), task.coaster.get_name()))

    checks = {
        "xy": xy_error < XY_MAX_M,
        "z": z_error < Z_MAX_M,
        "tilt": tilt_deg <= TILT_MAX_DEG,
        "linear_speed": linear_speed is not None and linear_speed <= LINEAR_SPEED_MAX_MPS,
        "angular_speed": angular_speed is not None and angular_speed <= ANGULAR_SPEED_MAX_RADPS,
        "cup_coaster_contact": contact,
    }
    return {
        "policy": {
            "xy_max_m_strict": XY_MAX_M,
            "z_max_m_strict": Z_MAX_M,
            "tilt_max_deg": TILT_MAX_DEG,
            "linear_speed_max_mps": LINEAR_SPEED_MAX_MPS,
            "angular_speed_max_radps": ANGULAR_SPEED_MAX_RADPS,
            "require_cup_coaster_contact": True,
            "upright_reference": "setup_demo_initial_pose_dominant_local_axis",
        },
        "measurements": {
            "xy_error_m": xy_error,
            "z_error_m": z_error,
            "tilt_deg": tilt_deg,
            "linear_speed_mps": linear_speed,
            "angular_speed_radps": angular_speed,
            "cup_coaster_contact": contact,
            "upright_local_axis_index": local_axis_index,
            "upright_axis_sign": axis_sign,
            "initial_tilt_deg": float(reference["initial_tilt_deg"]),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    robotwin_root = args.robotwin_root.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc).astimezone().isoformat()
    with _robotwin_import_context(robotwin_root):
        task_module = importlib.import_module(f"envs.{TASK_NAME}")
        task_class = getattr(task_module, TASK_NAME)
        base_args = _load_base_args(robotwin_root, output_dir)

        def factory(_seed: int) -> Any:
            return task_class()

        try:
            plan = search_accepted_plan(
                factory,
                base_args=base_args,
                start_seed=args.start_seed,
                max_attempts=args.max_attempts,
                trajectory_index=0,
            )
        except PilotQuotaError as exc:
            failure = {
                "status": "BOUNDED_PLANNING_EXHAUSTED",
                "task": TASK_NAME,
                "started_at": started,
                "finished_at": datetime.now(timezone.utc).astimezone().isoformat(),
                "max_attempts": args.max_attempts,
                "attempts": [attempt.to_dict() for attempt in exc.attempts],
                "accepted": False,
            }
            _write_json(output_dir / "acceptance_receipt.json", failure)
            return 2

        replay = replay_accepted_plan(
            factory,
            base_args=base_args,
            accepted_seed=plan.accepted_seed,
            trajectory_index=0,
            instructions=["place the empty cup on the coaster"],
            pre_replay_probe=_capture_upright_reference,
            quality_probe=_quality_probe,
        )

    hdf5_path = output_dir / "data/episode_0000000.hdf5"
    if not hdf5_path.is_file() or hdf5_path.stat().st_size == 0:
        raise FileNotFoundError(f"expected non-empty HDF5 output: {hdf5_path}")
    (output_dir / "seed.txt").write_text(f"{plan.accepted_seed}\n", encoding="utf-8")
    _write_json(output_dir / "scene_info.json", {"episode_0": dict(replay.episode_info)})

    receipt = {
        "status": "ORACLE_SUCCESS_QUALITY_PASS" if replay.quality.get("passed") else "ORACLE_SUCCESS_QUALITY_HOLD",
        "task": TASK_NAME,
        "provenance_class": "online scripted-expert planning + saved joint-path replay collection",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "gina_root": os.fspath(args.gina_root.resolve(strict=True)),
        "gina_head": _git_head(args.gina_root),
        "robotwin_root": os.fspath(robotwin_root),
        "robotwin_head": _git_head(robotwin_root),
        "output_dir": os.fspath(output_dir),
        "plan": plan.to_dict(),
        "replay": replay.to_dict(),
        "artifact": {
            "path": os.fspath(hdf5_path),
            "bytes": hdf5_path.stat().st_size,
            "sha256": _sha256(hdf5_path),
        },
        "native_oracle_success": True,
        "scale_up_eligible": bool(replay.quality.get("passed")),
        "training_manifest_eligible": False,
        "note": "Training eligibility remains false until independent HDF5/schema/modality readback passes.",
    }
    _write_json(output_dir / "acceptance_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["scale_up_eligible"] else 3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gina-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=25)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
