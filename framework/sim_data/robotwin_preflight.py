#!/usr/bin/env python3
"""Read-only fail-closed preflight for a RoboTwin ``place_empty_cup`` pilot.

The checker reads paths and source, asks Git for HEAD, and uses
``importlib.util.find_spec`` without importing packages.  It never installs,
downloads, writes into RoboTwin, starts a simulator, or performs a rollout.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping

PINNED_ROBOTWIN_COMMIT = "ea8b21121ebb3cd201ff5b3fe361944ac94eda3f"
DEFAULT_ROBOTWIN_ROOT = Path("/home/ai-sz-a26394-u1/RoboTwin")
DEFAULT_MAX_ATTEMPTS = 25

# Import names, not distribution names.  This is the minimum native collector
# runtime surface checked before any attempt to import RoboTwin itself.
REQUIRED_PACKAGES = (
    "torch",
    "torchvision",
    "numpy",
    "cv2",
    "transforms3d",
    "sapien",
    "scipy",
    "mplib",
    "gymnasium",
    "trimesh",
    "open3d",
    "imageio",
    "pydantic",
    "yaml",
    "zarr",
    "h5py",
)

ASSET_REQUIREMENTS = {
    "cup": "assets/objects/021_cup",
    "coaster": "assets/objects/019_coaster",
    "embodiment": "assets/embodiments/aloha-agilex",
    "background_texture": "assets/background_texture",
}

CommitReader = Callable[[Path], str]
PackageFinder = Callable[[str], object | None]


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "details": dict(self.details)}


@dataclass(frozen=True)
class PreflightReport:
    checks: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(check.ok for check in self.checks)

    def by_name(self, name: str) -> CheckResult:
        matches = [check for check in self.checks if check.name == name]
        if len(matches) != 1:
            raise KeyError(f"expected exactly one check named {name!r}, found {len(matches)}")
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": [check.to_dict() for check in self.checks]}


@dataclass(frozen=True)
class PreflightInputs:
    project_root: Path
    expected_root: Path
    expected_commit: str
    environment: Mapping[str, str]
    pilot_config: Mapping[str, Any]
    commit_reader: CommitReader
    package_finder: PackageFinder


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", os.fspath(root), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip()


def _find_package(name: str) -> object | None:
    return importlib.util.find_spec(name)


def _resolved(path: Path) -> tuple[Path | None, str | None]:
    try:
        return path.expanduser().resolve(strict=True), None
    except (OSError, RuntimeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _source_tree(path: Path) -> tuple[ast.AST | None, str | None]:
    try:
        source = path.read_text(encoding="utf-8")
        if not source.strip():
            return None, "source is empty"
        return ast.parse(source, filename=os.fspath(path)), None
    except (OSError, UnicodeError, SyntaxError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _attribute_names(node: ast.AST) -> set[str]:
    return {child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute)}


def _function_gate_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    return _attribute_names(function)


def _collector_has_direct_combined_gate(tree: ast.AST) -> bool:
    required = {"plan_success", "check_success", "planned_joints_legal"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.Return)):
            expression = node.test if isinstance(node, ast.If) else node.value
            if expression is not None and required <= _attribute_names(expression):
                return True
    return False


def _collector_has_nested_combined_gate(tree: ast.AST) -> bool:
    """Recognize upstream's outer plan/task gate plus inner joint gate."""
    for outer in (node for node in ast.walk(tree) if isinstance(node, ast.If)):
        if not {"plan_success", "check_success"} <= _attribute_names(outer.test):
            continue
        for inner in (node for statement in outer.body for node in ast.walk(statement)):
            if not isinstance(inner, ast.If) or not inner.orelse:
                continue
            assigned_joint_names: set[str] = set()
            for statement in outer.body:
                if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                    continue
                value = statement.value
                if value is None or "planned_joints_legal" not in _attribute_names(value):
                    continue
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                for target in targets:
                    assigned_joint_names.update(
                        child.id for child in ast.walk(target) if isinstance(child, ast.Name)
                    )
            tested_names = {
                child.id for child in ast.walk(inner.test) if isinstance(child, ast.Name)
            }
            if assigned_joint_names & tested_names:
                return True
    return False


def _nonempty_regular_file_count(path: Path) -> tuple[int, str | None]:
    try:
        if not path.is_dir():
            return 0, "required asset directory is missing"
        count = sum(
            1
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.stat().st_size > 0
        )
        return count, None if count else "no non-empty regular file found"
    except OSError as exc:
        return 0, f"{type(exc).__name__}: {exc}"


def check_preflight(inputs: PreflightInputs) -> PreflightReport:
    """Evaluate every gate independently and return a fail-closed report."""
    checks: list[CheckResult] = []

    actual_root, actual_error = _resolved(inputs.project_root)
    expected_root, expected_error = _resolved(inputs.expected_root)
    root_ok = (
        actual_error is None
        and expected_error is None
        and actual_root is not None
        and actual_root == expected_root
    )
    checks.append(
        CheckResult(
            "project_root",
            root_ok,
            {
                "actual": os.fspath(actual_root) if actual_root else None,
                "expected": os.fspath(expected_root) if expected_root else None,
                "error": actual_error or expected_error,
            },
        )
    )

    observed_commit: str | None = None
    commit_error: str | None = None
    if actual_root is not None:
        try:
            observed_commit = inputs.commit_reader(actual_root).strip()
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            commit_error = f"{type(exc).__name__}: {exc}"
    else:
        commit_error = "project root is unavailable"
    checks.append(
        CheckResult(
            "git_commit",
            observed_commit == inputs.expected_commit,
            {
                "actual": observed_commit,
                "expected": inputs.expected_commit,
                "error": commit_error,
            },
        )
    )

    no_user_site = inputs.environment.get("PYTHONNOUSERSITE")
    checks.append(
        CheckResult(
            "python_no_user_site",
            no_user_site is not None and no_user_site != "",
            {
                "present": no_user_site is not None,
                "nonempty": bool(no_user_site),
                "semantic": "CPython disables user site for any non-empty value",
            },
        )
    )

    missing_packages: list[str] = []
    package_errors: dict[str, str] = {}
    for package in REQUIRED_PACKAGES:
        try:
            if inputs.package_finder(package) is None:
                missing_packages.append(package)
        except (ImportError, AttributeError, ValueError) as exc:
            missing_packages.append(package)
            package_errors[package] = f"{type(exc).__name__}: {exc}"
    checks.append(
        CheckResult(
            "required_packages",
            not missing_packages,
            {
                "required": list(REQUIRED_PACKAGES),
                "missing": missing_packages,
                "errors": package_errors,
            },
        )
    )

    asset_root = actual_root if actual_root is not None else inputs.project_root
    for label, relative in ASSET_REQUIREMENTS.items():
        asset_path = asset_root / relative
        count, error = _nonempty_regular_file_count(asset_path)
        required = not (
            label == "background_texture"
            and inputs.pilot_config.get("random_background", True) is False
        )
        checks.append(
            CheckResult(
                f"asset:{label}",
                (count > 0 and error is None) if required else True,
                {
                    "path": os.fspath(asset_path),
                    "required": required,
                    "nonempty_file_count": count,
                    "error": error,
                },
            )
        )

    task_path = asset_root / "envs/place_empty_cup.py"
    task_tree, task_error = _source_tree(task_path)
    methods: set[str] = set()
    if task_tree is not None:
        for node in ast.walk(task_tree):
            if isinstance(node, ast.ClassDef) and node.name == "place_empty_cup":
                methods.update(
                    child.name
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                )
    task_required = {"play_once", "check_success"}
    checks.append(
        CheckResult(
            "task_source_contract",
            task_tree is not None and task_required <= methods,
            {
                "path": os.fspath(task_path),
                "required_methods": sorted(task_required),
                "found_methods": sorted(methods),
                "error": task_error,
            },
        )
    )

    collector_path = asset_root / "scripts/collect_data.py"
    collector_tree, collector_error = _source_tree(collector_path)
    collector_functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    if collector_tree is not None:
        collector_functions = [
            node
            for node in ast.walk(collector_tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
    required_gates = {"plan_success", "check_success", "planned_joints_legal"}
    present_gates = set().union(
        *(_function_gate_names(function) for function in collector_functions), set()
    )
    combined_gate = bool(
        collector_tree is not None
        and (
            _collector_has_direct_combined_gate(collector_tree)
            or _collector_has_nested_combined_gate(collector_tree)
        )
    )
    checks.append(
        CheckResult(
            "collector_native_gate",
            required_gates <= present_gates and combined_gate,
            {
                "path": os.fspath(collector_path),
                "required": sorted(required_gates),
                "present": sorted(required_gates & present_gates),
                "combined_admission": combined_gate,
                "error": collector_error,
            },
        )
    )

    episode_num = inputs.pilot_config.get("episode_num")
    max_attempts = inputs.pilot_config.get("max_attempts")
    episode_ok = type(episode_num) is int and episode_num == 1
    attempts_ok = type(max_attempts) is int and max_attempts > 0
    checks.append(
        CheckResult(
            "bounded_pilot_config",
            episode_ok and attempts_ok,
            {
                "episode_num": episode_num,
                "max_attempts": max_attempts,
                "requirement": "episode_num == 1 and max_attempts is a positive integer",
            },
        )
    )

    return PreflightReport(tuple(checks))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--expected-root", type=Path, default=DEFAULT_ROBOTWIN_ROOT)
    parser.add_argument("--expected-commit", default=PINNED_ROBOTWIN_COMMIT)
    parser.add_argument("--episode-num", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument(
        "--random-background",
        action="store_true",
        help="require the background texture asset used by randomized backgrounds",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = PreflightInputs(
        project_root=args.project_root,
        expected_root=args.expected_root,
        expected_commit=args.expected_commit,
        environment=os.environ,
        pilot_config={
            "episode_num": args.episode_num,
            "max_attempts": args.max_attempts,
            "random_background": args.random_background,
        },
        commit_reader=_git_head,
        package_finder=_find_package,
    )
    report = check_preflight(inputs)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
