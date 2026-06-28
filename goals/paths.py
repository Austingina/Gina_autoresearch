"""AutoResearch workspace paths — all mutable data stays under project root."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Read-only upstream repos (never write datasets here from this project).
ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_ROOT", "/home/l/code/IsaacLab"))
ISAAC_DATASETS = ISAACLAB_ROOT / "datasets"


def workspace_root() -> Path:
    return ROOT


def data_root() -> Path:
    return ROOT / "data"


def task_data_dir(task: str, stage: str) -> Path:
    return data_root() / task / stage / "demos"


def manifests_dir() -> Path:
    return data_root() / "manifests"


def checkpoints_dir() -> Path:
    return ROOT / "checkpoints"


def runs_dir() -> Path:
    return ROOT / "runs"


def field_checkpoint(task: str, stage: str) -> Path:
    return checkpoints_dir() / task / f"guidance_field_{stage}.pt"


def default_demo_dir(task: str = "lift", stage: str = "s0_fixed") -> Path:
    return task_data_dir(task, stage)


def is_under_isaac_datasets(path: Path) -> bool:
    try:
        path.resolve().relative_to(ISAAC_DATASETS.resolve())
        return True
    except ValueError:
        return False


def is_under_workspace(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT.resolve())
        return True
    except ValueError:
        return False


def assert_writable_data_path(path: Path, *, allow_external: bool = False) -> None:
    """Refuse paths that would read/write IsaacLab canonical datasets in-place."""
    resolved = path.resolve()
    if not allow_external and is_under_isaac_datasets(resolved):
        raise ValueError(
            f"Refusing IsaacLab dataset path (read-only for this project): {resolved}\n"
            f"  Copy first: python scripts/import_isaac_demos.py --source ... --stage s0_fixed\n"
            f"  Or pass --allow-external-data to override (not recommended)."
        )


def ensure_workspace_dirs(task: str = "lift", stage: str = "s0_fixed") -> None:
    for d in (
        task_data_dir(task, stage),
        manifests_dir(),
        checkpoints_dir() / task,
        runs_dir(),
    ):
        d.mkdir(parents=True, exist_ok=True)
