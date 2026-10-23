#!/usr/bin/env python3
"""Isolated launcher and CPU-only preflight for Gina on Isaac Sim 6.0.1.

This file never initializes Kit during ``--cpu-preflight``.  Normal command mode
executes the standalone runtime's ``python.sh`` with Gina's vendored IsaacLab
source roots first and a Python-3.12 dependency interpreter.  It deliberately
removes active conda/venv and inherited PYTHONPATH state so Isaac Sim 5.1 or a
separately installed IsaacLab cannot enter module resolution.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_ROOT = PROJECT_ROOT / "simulators" / "isaaclab"
ISAACSIM_ROOT = Path("/home/ai-sz-a26394-u1/isaacsim/6.0.1")
DEPENDENCY_PYTHON = Path("/home/ai-sz-a26394-u1/miniforge3/envs/env_isaaclab/bin/python")
DEPENDENCY_LIB = DEPENDENCY_PYTHON.parent.parent / "lib"
RUNTIME_PYTHON_SH = ISAACSIM_ROOT / "python.sh"

VENDORED_SOURCE_PATHS = tuple(
    ISAACLAB_ROOT / "source" / name
    for name in (
        "isaaclab",
        "isaaclab_assets",
        "isaaclab_tasks",
        "isaaclab_mimic",
        "isaaclab_rl",
        "isaaclab_contrib",
    )
)

_SCRUBBED_ENV_KEYS = {
    "CONDA_DEFAULT_ENV",
    "CONDA_EXE",
    "CONDA_PREFIX",
    "CONDA_PROMPT_MODIFIER",
    "CONDA_PYTHON_EXE",
    "CONDA_SHLVL",
    "PYTHONHOME",
    "PYTHONPATH",
    "VIRTUAL_ENV",
}


def build_launch_environment(parent: Mapping[str, str]) -> dict[str, str]:
    """Return an isolated environment without inherited Python/runtime paths."""
    env = {key: value for key, value in parent.items() if key not in _SCRUBBED_ENV_KEYS}
    inherited_library_path = [part for part in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part]
    env["LD_LIBRARY_PATH"] = os.pathsep.join([str(DEPENDENCY_LIB), *inherited_library_path])
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONEXE": str(DEPENDENCY_PYTHON),
            "PYTHONPATH": os.pathsep.join(str(path) for path in VENDORED_SOURCE_PATHS),
            "ISAACSIM_ROOT": str(ISAACSIM_ROOT),
            "ISAACLAB_ROOT": str(ISAACLAB_ROOT),
        }
    )
    return env


def _top_level_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _python_version(executable: Path) -> list[int]:
    completed = subprocess.run(
        [str(executable), "-c", "import sys; print(*sys.version_info[:3])"],
        check=True,
        text=True,
        capture_output=True,
        env={"PATH": os.environ.get("PATH", "/usr/bin"), "PYTHONNOUSERSITE": "1"},
    )
    return [int(part) for part in completed.stdout.strip().split()]


def _runtime_python_version() -> list[int]:
    completed = subprocess.run(
        [str(RUNTIME_PYTHON_SH), "-c", "import sys; print(*sys.version_info[:3])"],
        check=True,
        text=True,
        capture_output=True,
        env={"PATH": os.environ.get("PATH", "/usr/bin"), "PYTHONNOUSERSITE": "1"},
    )
    fields = completed.stdout.strip().splitlines()[-1].split()
    return [int(part) for part in fields]


def cpu_preflight() -> dict[str, object]:
    """Statically audit versions and bundled APIs; never import or start Kit."""
    required_paths = [ISAACSIM_ROOT, ISAACLAB_ROOT, DEPENDENCY_PYTHON, RUNTIME_PYTHON_SH, *VENDORED_SOURCE_PATHS]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required Isaac 6 launch paths: " + ", ".join(missing))

    sim_version = (ISAACSIM_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    lab_release = (ISAACLAB_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    with (ISAACLAB_ROOT / "source" / "isaaclab" / "config" / "extension.toml").open("rb") as stream:
        lab_core_version = tomllib.load(stream)["package"]["version"]

    command_source = (
        ISAACSIM_ROOT
        / "extscache"
        / "omni.kit.commands-1.5.0+f9bf0dda.lx64.r.cp312"
        / "omni"
        / "kit"
        / "commands"
        / "command.py"
    )
    usd_stub = (
        ISAACSIM_ROOT
        / "extscache"
        / "omni.usd-1.16.0+f9bf0dda.lx64.r.cp312"
        / "omni"
        / "usd"
        / "_usd.pyi"
    )
    command_functions = _top_level_functions(command_source)
    usd_functions = _top_level_functions(usd_stub)
    dependency_version = _python_version(DEPENDENCY_PYTHON)
    runtime_version = _runtime_python_version()

    api_contracts = {
        "omni.kit.commands.register_callback": "register_callback" in command_functions,
        "omni.kit.commands.unregister_module_commands": "unregister_module_commands" in command_functions,
        "omni.usd.get_context": "get_context" in usd_functions,
    }
    checks = {
        "isaac_sim_6_0_1": sim_version.startswith("6.0.1"),
        "isaac_lab_2_3_2": lab_release == "2.3.2",
        "python_abi_match": dependency_version[:2] == runtime_version[:2] == [3, 12],
        "api_contracts": all(api_contracts.values()),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "kit_started": False,
        "isaac_sim_root": str(ISAACSIM_ROOT),
        "isaac_sim_version": sim_version,
        "isaac_lab_root": str(ISAACLAB_ROOT),
        "isaac_lab_release": lab_release,
        "isaac_lab_core_version": lab_core_version,
        "dependency_python": str(DEPENDENCY_PYTHON),
        "dependency_python_version": dependency_version,
        "runtime_python_version": runtime_version,
        "vendored_source_paths": [str(path) for path in VENDORED_SOURCE_PATHS],
        "api_sources": {
            "omni.kit.commands": str(command_source),
            "omni.usd": str(usd_stub),
        },
        "api_contracts": api_contracts,
        "checks": checks,
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-preflight", action="store_true", help="Print CPU-only JSON audit and do not start Kit.")
    parser.add_argument("--print-command", action="store_true", help="Print the isolated command instead of executing it.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Python script/module arguments; prefix with --.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.cpu_preflight:
        receipt = cpu_preflight()
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "PASS" else 1

    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise SystemExit("a Python command is required after -- (or use --cpu-preflight)")

    receipt = cpu_preflight()
    if receipt["status"] != "PASS":
        print(json.dumps(receipt, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    launch = [str(RUNTIME_PYTHON_SH), *command]
    if args.print_command:
        print(json.dumps({"argv": launch, "env": build_launch_environment(os.environ)}, indent=2, sort_keys=True))
        return 0
    os.execve(str(RUNTIME_PYTHON_SH), launch, build_launch_environment(os.environ))
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
