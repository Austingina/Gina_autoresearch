"""Native success-gate adapters for CALVIN, LIBERO, and RoboTwin."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .contracts import OracleEvidence


class NativeOracleError(RuntimeError):
    """Raised when a simulator's authoritative success API is unavailable."""


def _explicit_bool(value: Any, *, source: str) -> bool:
    if isinstance(value, bool):
        return value
    if value.__class__.__name__ == "bool_" and value.__class__.__module__.startswith("numpy"):
        return bool(value)
    raise TypeError(f"{source} must return an explicit boolean, got {type(value)!r}")


def calvin_oracle_gate(oracle: Any, initial_info: Any, final_info: Any, task: str) -> OracleEvidence:
    checker = getattr(oracle, "get_task_info_for_set", None)
    if not callable(checker):
        raise NativeOracleError("CALVIN oracle lacks get_task_info_for_set")
    achieved = checker(initial_info, final_info, {task})
    if achieved is None or isinstance(achieved, (str, bytes)) or not isinstance(achieved, Iterable):
        raise NativeOracleError("CALVIN oracle returned no iterable achieved-task set")
    achieved_tasks = sorted(str(item) for item in achieved)
    return OracleEvidence(
        simulator="calvin",
        task=task,
        checker="get_task_info_for_set(initial_info, final_info, {task})",
        success=task in achieved_tasks,
        details={"achieved_tasks": achieved_tasks, "same_rollout_initial_final": True},
    )


def libero_oracle_gate(env: Any, task: str) -> OracleEvidence:
    checker = getattr(env, "check_success", None)
    if not callable(checker):
        raise NativeOracleError("LIBERO environment lacks public check_success()")
    success = _explicit_bool(checker(), source="LIBERO check_success()")
    return OracleEvidence(
        simulator="libero",
        task=task,
        checker="ControlEnv.check_success() -> task _check_success()",
        success=success,
        details={"native_done_predicate": True},
    )


def robotwin_oracle_gate(task_env: Any, task: str) -> OracleEvidence:
    if not hasattr(task_env, "plan_success"):
        raise NativeOracleError("RoboTwin task lacks plan_success")
    checker = getattr(task_env, "check_success", None)
    joint_checker = getattr(task_env, "planned_joints_legal", None)
    if not callable(checker) or not callable(joint_checker):
        raise NativeOracleError("RoboTwin task lacks check_success() or planned_joints_legal()")
    plan_success = _explicit_bool(task_env.plan_success, source="RoboTwin plan_success")
    task_success = _explicit_bool(checker(), source="RoboTwin check_success()")
    joint_result = joint_checker()
    if not isinstance(joint_result, tuple) or len(joint_result) != 2:
        raise TypeError("RoboTwin planned_joints_legal() must return (bool, excess)")
    joints_legal = _explicit_bool(joint_result[0], source="RoboTwin planned_joints_legal()[0]")
    success = bool(plan_success and task_success and joints_legal)
    return OracleEvidence(
        simulator="robotwin",
        task=task,
        checker="plan_success && check_success() && planned_joints_legal()",
        success=success,
        details={
            "plan_success": plan_success,
            "task_success": task_success,
            "planned_joints_legal": joints_legal,
            "joint_excess_rad": float(joint_result[1]),
        },
    )
