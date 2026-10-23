#!/usr/bin/env python3
"""Bounded, fail-closed adapter for RoboTwin native expert pilots.

The module keeps RoboTwin source read-only.  Planning attempts are isolated in
fresh task instances and an accepted joint path requires the native conjunction
``plan_success && check_success() && planned_joints_legal()``.  The CLI/replay
layer is added only after these bounded-search contracts are green.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence


QualityProbe = Callable[[Any], Mapping[str, Any]]
PreReplayProbe = Callable[[Any], None]


@dataclass(frozen=True)
class AttemptReceipt:
    seed: int
    plan_success: bool
    oracle_success: bool
    joints_legal: bool
    joint_excess_rad: float | None
    accepted: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanSearchResult:
    accepted_seed: int
    trajectory_index: int
    attempts: tuple[AttemptReceipt, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_seed": self.accepted_seed,
            "trajectory_index": self.trajectory_index,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


@dataclass(frozen=True)
class ReplayResult:
    seed: int
    trajectory_index: int
    oracle_success: bool
    episode_info: Mapping[str, Any]
    quality: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "trajectory_index": self.trajectory_index,
            "oracle_success": self.oracle_success,
            "episode_info": dict(self.episode_info),
            "quality": dict(self.quality),
        }


class PilotQuotaError(RuntimeError):
    def __init__(self, attempts: Sequence[AttemptReceipt]):
        self.attempts = tuple(attempts)
        super().__init__(
            f"no accepted RoboTwin plan within {len(self.attempts)} bounded attempts"
        )


class ReplayOracleError(RuntimeError):
    pass


TaskFactory = Callable[[int], Any]


def _close_task(task: Any) -> None:
    close = getattr(task, "close_env", None)
    if callable(close):
        close()


def search_accepted_plan(
    task_factory: TaskFactory,
    *,
    base_args: Mapping[str, Any],
    start_seed: int,
    max_attempts: int,
    trajectory_index: int,
) -> PlanSearchResult:
    """Search a bounded seed range and persist only a native-gate-valid plan."""
    if type(max_attempts) is not int or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")
    if type(start_seed) is not int:
        raise ValueError("start_seed must be an integer")

    attempts: list[AttemptReceipt] = []
    for offset in range(max_attempts):
        seed = start_seed + offset
        task = task_factory(seed)
        try:
            task.setup_demo(
                now_ep_num=trajectory_index,
                seed=seed,
                **dict(base_args),
            )
            task.play_once()
            plan_success = bool(task.plan_success)
            oracle_success = bool(task.check_success()) if plan_success else False
            if plan_success and oracle_success:
                joints_legal, excess = task.planned_joints_legal()
                joints_legal = bool(joints_legal)
                joint_excess = float(excess)
            else:
                joints_legal = False
                joint_excess = None
            accepted = plan_success and oracle_success and joints_legal
            receipt = AttemptReceipt(
                seed=seed,
                plan_success=plan_success,
                oracle_success=oracle_success,
                joints_legal=joints_legal,
                joint_excess_rad=joint_excess,
                accepted=accepted,
            )
            attempts.append(receipt)
            if accepted:
                task.save_traj_data(trajectory_index)
                return PlanSearchResult(seed, trajectory_index, tuple(attempts))
        except Exception as exc:
            attempts.append(
                AttemptReceipt(
                    seed=seed,
                    plan_success=False,
                    oracle_success=False,
                    joints_legal=False,
                    joint_excess_rad=None,
                    accepted=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        finally:
            _close_task(task)

    raise PilotQuotaError(attempts)


def replay_accepted_plan(
    task_factory: TaskFactory,
    *,
    base_args: Mapping[str, Any],
    accepted_seed: int,
    trajectory_index: int,
    instructions: Sequence[str],
    pre_replay_probe: PreReplayProbe | None = None,
    quality_probe: QualityProbe | None = None,
) -> ReplayResult:
    """Replay a saved plan and write payload only after native Oracle success."""
    task = task_factory(accepted_seed)
    replay_args = dict(base_args)
    replay_args.update({"need_plan": False, "save_data": True, "render_freq": 0})
    episode_info: Mapping[str, Any] = {}
    quality: Mapping[str, Any] = {}
    oracle_success = False
    try:
        task.setup_demo(
            now_ep_num=trajectory_index,
            seed=accepted_seed,
            **replay_args,
        )
        if pre_replay_probe is not None:
            pre_replay_probe(task)
        trajectory = task.load_tran_data(trajectory_index)
        replay_args["left_joint_path"] = trajectory["left_joint_path"]
        replay_args["right_joint_path"] = trajectory["right_joint_path"]
        task.set_path_lst(replay_args)
        raw_info = task.play_once()
        if isinstance(raw_info, Mapping):
            episode_info = raw_info
        oracle_success = bool(task.check_success())
        if not oracle_success:
            raise ReplayOracleError(
                f"RoboTwin replay failed native Oracle for seed {accepted_seed}"
            )
        if quality_probe is not None:
            quality = dict(quality_probe(task))
    finally:
        _close_task(task)

    # This ordering is intentional: unlike upstream collect_data.py, a failed
    # replay cannot leave a trajectory file that looks accepted by existence.
    task.merge_pkl_to_hdf5_video(instructions=list(instructions))
    task.remove_data_cache()
    return ReplayResult(
        seed=accepted_seed,
        trajectory_index=trajectory_index,
        oracle_success=oracle_success,
        episode_info=episode_info,
        quality=quality,
    )
