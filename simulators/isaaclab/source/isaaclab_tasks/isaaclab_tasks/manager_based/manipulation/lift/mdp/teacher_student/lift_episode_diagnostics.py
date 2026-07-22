# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Per-episode lift diagnostics: trajectories + failure-mode taxonomy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class LiftFailureMode(str, Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    TERMINATED = "terminated"
    NEVER_REACHED_CUBE = "never_reached_cube"
    NEVER_GRASPED = "never_grasped"
    GRASP_THEN_DROPPED = "grasp_then_dropped"
    LIFTED_NOT_AT_GOAL = "lifted_not_at_goal"
    OTHER = "other"


AUX_SUCCESS_THRESHOLD_M = 0.05
TABLE_Z_M = 0.04


@dataclass
class StepSnapshot:
    step: int
    ee_x: float
    ee_y: float
    ee_z: float
    cube_x: float
    cube_y: float
    cube_z: float
    goal_x: float
    goal_y: float
    goal_z: float
    dist_to_goal: float
    ee_cube_dist: float
    gripper_open: float
    gripper_cmd: float
    teacher_phase: int


@dataclass
class EpisodeDiagnostic:
    episode_id: int
    steps: int
    end_reason: str
    failure_mode: str
    ever_grasp: bool
    ever_lift_success: bool
    max_cube_z: float
    final_cube_z: float
    min_dist_to_goal: float
    final_dist_to_goal: float
    min_ee_cube_dist: float
    pre_grasp_min_ee_cube: float
    post_grasp_dist_goal: float
    post_grasp_wander_m: float
    first_grasp_step: int | None
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def episode_reached_goal(ep: EpisodeDiagnostic, threshold_m: float) -> bool:
    """Episode-level lift success at distance threshold (min dist + cube lifted)."""
    return ep.max_cube_z > TABLE_Z_M + 0.01 and ep.min_dist_to_goal <= threshold_m


class EpisodeDiagnosticRecorder:
    """Accumulate step-wise EE/cube/goal state for one env slot."""

    TABLE_Z = 0.04
    EE_GRASP_DIST = 0.07
    REACH_CUBE_DIST = 0.10
    GOAL_THRESH = 0.02

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._steps: list[StepSnapshot] = []
        self._ever_grasp = False
        self._ever_lift = False

    def record_step(
        self,
        *,
        step: int,
        ee_pos_b: np.ndarray,
        cube_pos_b: np.ndarray,
        goal_pos_b: np.ndarray,
        dist_to_goal: float,
        ee_cube_dist: float,
        gripper_open: float,
        gripper_cmd: float,
        teacher_phase: int,
        grasp_now: bool,
        lift_now: bool,
    ) -> None:
        self._ever_grasp |= bool(grasp_now)
        self._ever_lift |= bool(lift_now)
        self._steps.append(
            StepSnapshot(
                step=step,
                ee_x=float(ee_pos_b[0]),
                ee_y=float(ee_pos_b[1]),
                ee_z=float(ee_pos_b[2]),
                cube_x=float(cube_pos_b[0]),
                cube_y=float(cube_pos_b[1]),
                cube_z=float(cube_pos_b[2]),
                goal_x=float(goal_pos_b[0]),
                goal_y=float(goal_pos_b[1]),
                goal_z=float(goal_pos_b[2]),
                dist_to_goal=float(dist_to_goal),
                ee_cube_dist=float(ee_cube_dist),
                gripper_open=float(gripper_open),
                gripper_cmd=float(gripper_cmd),
                teacher_phase=int(teacher_phase),
            )
        )

    def finalize(
        self,
        *,
        episode_id: int,
        end_reason: str,
        ever_grasp_flag: bool,
        ever_lift_flag: bool,
        save_trajectory: bool = True,
        trajectory_stride: int = 5,
    ) -> EpisodeDiagnostic:
        if not self._steps:
            return EpisodeDiagnostic(
                episode_id=episode_id,
                steps=0,
                end_reason=end_reason,
                failure_mode=LiftFailureMode.OTHER.value,
                ever_grasp=False,
                ever_lift_success=False,
                max_cube_z=0.0,
                final_cube_z=0.0,
                min_dist_to_goal=999.0,
                final_dist_to_goal=999.0,
                min_ee_cube_dist=999.0,
                pre_grasp_min_ee_cube=999.0,
                post_grasp_dist_goal=999.0,
                post_grasp_wander_m=0.0,
                first_grasp_step=None,
            )

        cube_z = np.array([s.cube_z for s in self._steps])
        dist_goal = np.array([s.dist_to_goal for s in self._steps])
        ee_cube = np.array([s.ee_cube_dist for s in self._steps])

        max_z = float(cube_z.max())
        final_z = float(cube_z[-1])
        min_dg = float(dist_goal.min())
        final_dg = float(dist_goal[-1])
        min_ee = float(ee_cube.min())

        first_grasp = _first_grasp_index(self._steps, self.TABLE_Z, self.EE_GRASP_DIST)
        wander = _post_grasp_wander_m(self._steps, first_grasp)
        pre_grasp_min = _pre_grasp_min_ee_cube_m(self._steps, first_grasp)
        post_grasp_dg = _post_grasp_dist_goal_m(self._steps, first_grasp)

        mode = classify_failure_mode(
            end_reason=end_reason,
            ever_lift=ever_lift_flag or self._ever_lift,
            ever_grasp=ever_grasp_flag or self._ever_grasp,
            max_cube_z=max_z,
            final_cube_z=final_z,
            final_dist_to_goal=final_dg,
            min_ee_cube_dist=min_ee,
            goal_threshold=self.GOAL_THRESH,
            table_z=self.TABLE_Z,
            reach_cube_dist=self.REACH_CUBE_DIST,
        )

        traj_out: list[dict[str, Any]] = []
        if save_trajectory:
            for i, s in enumerate(self._steps):
                if i % trajectory_stride == 0 or i == len(self._steps) - 1:
                    traj_out.append(asdict(s))

        return EpisodeDiagnostic(
            episode_id=episode_id,
            steps=len(self._steps),
            end_reason=end_reason,
            failure_mode=mode.value,
            ever_grasp=bool(ever_grasp_flag or self._ever_grasp),
            ever_lift_success=bool(ever_lift_flag or self._ever_lift),
            max_cube_z=max_z,
            final_cube_z=final_z,
            min_dist_to_goal=min_dg,
            final_dist_to_goal=final_dg,
            min_ee_cube_dist=min_ee,
            pre_grasp_min_ee_cube=pre_grasp_min,
            post_grasp_dist_goal=post_grasp_dg,
            post_grasp_wander_m=wander,
            first_grasp_step=first_grasp,
            trajectory=traj_out,
            summary={
                "mean_ee_cube_dist": float(ee_cube.mean()),
                "final_gripper_open": float(self._steps[-1].gripper_open),
                "max_cube_z_step": int(cube_z.argmax()),
            },
        )


def _first_grasp_index(steps: list[StepSnapshot], table_z: float, ee_thresh: float) -> int | None:
    for s in steps:
        if s.cube_z > table_z and s.ee_cube_dist < ee_thresh:
            return s.step
    return None


def _pre_grasp_min_ee_cube_m(steps: list[StepSnapshot], first_grasp_step: int | None) -> float:
    """Minimum EE–cube distance before first grasp (approach quality)."""
    if first_grasp_step is None:
        return min((s.ee_cube_dist for s in steps), default=999.0)
    pre = [s.ee_cube_dist for s in steps if s.step < first_grasp_step]
    return min(pre) if pre else min((s.ee_cube_dist for s in steps), default=999.0)


def _post_grasp_dist_goal_m(steps: list[StepSnapshot], first_grasp_step: int | None) -> float:
    """Minimum cube–goal distance after first grasp (relocation quality)."""
    if first_grasp_step is None:
        return 999.0
    after = [s.dist_to_goal for s in steps if s.step >= first_grasp_step and s.cube_z > 0.04]
    return min(after) if after else 999.0


def _post_grasp_wander_m(steps: list[StepSnapshot], first_grasp_step: int | None) -> float:
    """Max horizontal cube–goal distance after first grasp (relocation wander)."""
    if first_grasp_step is None:
        return 0.0
    after = [s for s in steps if s.step >= first_grasp_step and s.cube_z > 0.04]
    if not after:
        return 0.0
    dists = [
        float(np.linalg.norm([s.cube_x - s.goal_x, s.cube_y - s.goal_y]))
        for s in after
    ]
    return max(dists) if dists else 0.0


def classify_failure_mode(
    *,
    end_reason: str,
    ever_lift: bool,
    ever_grasp: bool,
    max_cube_z: float,
    final_cube_z: float,
    final_dist_to_goal: float,
    min_ee_cube_dist: float,
    goal_threshold: float = 0.02,
    table_z: float = 0.04,
    reach_cube_dist: float = 0.10,
) -> LiftFailureMode:
    if ever_lift:
        return LiftFailureMode.SUCCESS
    if min_ee_cube_dist > reach_cube_dist:
        return LiftFailureMode.NEVER_REACHED_CUBE
    if max_cube_z < table_z + 0.01:
        return LiftFailureMode.NEVER_GRASPED
    if max_cube_z >= table_z and final_cube_z < table_z:
        return LiftFailureMode.GRASP_THEN_DROPPED
    if max_cube_z >= table_z and final_cube_z >= table_z and final_dist_to_goal > goal_threshold:
        return LiftFailureMode.LIFTED_NOT_AT_GOAL
    if end_reason == "timeout":
        return LiftFailureMode.TIMEOUT
    if end_reason == "terminated":
        return LiftFailureMode.TERMINATED
    return LiftFailureMode.OTHER


def aggregate_diagnostics(episodes: list[EpisodeDiagnostic]) -> dict[str, Any]:
    n = max(len(episodes), 1)
    modes: dict[str, int] = {}
    for ep in episodes:
        modes[ep.failure_mode] = modes.get(ep.failure_mode, 0) + 1
    success = modes.get(LiftFailureMode.SUCCESS.value, 0)
    success_5cm = sum(1 for ep in episodes if episode_reached_goal(ep, AUX_SUCCESS_THRESHOLD_M))
    wanders = [ep.post_grasp_wander_m for ep in episodes if ep.ever_grasp and not ep.ever_lift_success]
    pre_grasp = [ep.pre_grasp_min_ee_cube for ep in episodes]
    post_grasp_dg = [ep.post_grasp_dist_goal for ep in episodes if ep.ever_grasp]
    return {
        "num_episodes": len(episodes),
        "success_rate": success / n,
        "success_rate_5cm": success_5cm / n,
        "grasp_rate": sum(1 for e in episodes if e.ever_grasp) / n,
        "failure_mode_counts": modes,
        "mean_post_grasp_wander_m": float(np.mean(wanders)) if wanders else 0.0,
        "mean_pre_grasp_min_ee_cube": float(np.mean(pre_grasp)) if pre_grasp else 0.0,
        "mean_post_grasp_dist_goal": float(np.mean(post_grasp_dg)) if post_grasp_dg else 0.0,
        "mean_episode_steps": float(np.mean([e.steps for e in episodes])) if episodes else 0.0,
        "mean_min_ee_cube_dist": float(np.mean([e.min_ee_cube_dist for e in episodes])) if episodes else 0.0,
    }
