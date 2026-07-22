"""AutoResearch workspace paths — all mutable data stays under project root."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _default_isaaclab_root() -> Path:
    """Prefer bundled simulator; fall back to legacy sibling clones."""
    bundled = ROOT / "simulators" / "isaaclab"
    if (bundled / "isaaclab.sh").is_file():
        return bundled
    for legacy in (Path.home() / "code" / "IsaacLab", Path("/home/l/code/IsaacLab")):
        if (legacy / "isaaclab.sh").is_file():
            return legacy.resolve()
    return bundled


# Read-only upstream simulator checkout (never write datasets here from this project).
ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_ROOT", str(_default_isaaclab_root())))
ISAAC_DATASETS = ISAACLAB_ROOT / "datasets"

# Optional secondary platforms
MANISKILL_ROOT = Path(os.environ.get("MANISKILL_ROOT", str(ROOT / "simulators" / "maniskill")))
ARENA_ROOT = Path(os.environ.get("ARENA_ROOT", str(ROOT / "simulators" / "arena")))


def _default_calvin_g_root() -> Path:
    env = os.environ.get("CALVIN_G_ROOT")
    if env:
        return Path(env)
    h200 = Path("/mnt/nvme33n1/wuk_code/calvin_g")
    if h200.is_dir():
        return h200
    return ROOT / "phase1_calvin" / "calvin_g"


CALVIN_G_ROOT = _default_calvin_g_root()
CALVIN_CODE = Path(
    os.environ.get(
        "CALVIN_CODE",
        str(
            ROOT / "phase1_calvin" / "calvin_code"
            if (ROOT / "phase1_calvin" / "calvin_code").exists()
            else CALVIN_G_ROOT / "code" / "calvin"
        ),
    )
)
CALVIN_ENV = Path(
    os.environ.get(
        "CALVIN_ENV",
        str(
            ROOT / "phase1_calvin" / "envs" / "calvin_h200"
            if (ROOT / "phase1_calvin" / "envs" / "calvin_h200").exists()
            else CALVIN_G_ROOT / "envs" / "calvin_h200"
        ),
    )
)


def workspace_root() -> Path:
    return ROOT


def simulators_root() -> Path:
    return ROOT / "simulators"


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
        simulators_root(),
    ):
        d.mkdir(parents=True, exist_ok=True)
