# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim rollout evaluation from the offline training process."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SimEvalResult:
    success_rate: float
    grasp_rate: float
    mean_episode_length: float
    mean_final_cube_to_goal: float
    num_episodes: int

    @classmethod
    def from_dict(cls, data: dict) -> SimEvalResult:
        return cls(
            success_rate=float(data["lift_success_rate"]),
            grasp_rate=float(data["grasp_success_rate"]),
            mean_episode_length=float(data["mean_episode_length"]),
            mean_final_cube_to_goal=float(data["mean_final_cube_to_goal"]),
            num_episodes=int(data["num_episodes"]),
        )


def run_isaac_sim_eval(
    *,
    checkpoint_path: str,
    output_json: str,
    isaaclab_dir: str | Path,
    eval_episodes: int = 20,
    eval_task: str = "Isaac-Lift-Cube-Franka-Vision-MaskDepth-Play-v0",
    device: str = "cuda:0",
    episode_length_s: float = 30.0,
    position_threshold: float = 0.02,
    num_envs: int = 1,
    headless: bool = True,
    extra_args: list[str] | None = None,
) -> SimEvalResult:
    """Run ``play_lift_cube_offline_cql_lstm.py --metrics_only`` in a subprocess."""
    isaaclab_dir = Path(isaaclab_dir).resolve()
    isaaclab_sh = isaaclab_dir / "isaaclab.sh"
    play_script = "scripts/tools/play_lift_cube_offline_cql_lstm.py"
    if not isaaclab_sh.is_file():
        raise FileNotFoundError(f"isaaclab.sh not found: {isaaclab_sh}")

    output_json_abs = os.path.abspath(output_json)
    output_dir = os.path.dirname(output_json_abs)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if os.path.isfile(output_json_abs):
        os.remove(output_json_abs)

    cmd = [
        str(isaaclab_sh),
        "-p",
        play_script,
        "--checkpoint",
        os.path.abspath(checkpoint_path),
        "--task",
        eval_task,
        "--num_envs",
        str(num_envs),
        "--max_episodes",
        str(eval_episodes),
        "--episode_length_s",
        str(episode_length_s),
        "--position_threshold",
        str(position_threshold),
        "--device",
        device,
        "--metrics_only",
        "--output_json",
        output_json_abs,
    ]
    if headless:
        cmd.append("--headless")
    if extra_args:
        cmd.extend(extra_args)

    print("[INFO] Launching Isaac Sim evaluation:")
    print("  " + " ".join(cmd))

    completed = subprocess.run(
        cmd,
        cwd=str(isaaclab_dir),
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Isaac Sim evaluation failed with exit code {completed.returncode}")

    if not os.path.isfile(output_json_abs):
        raise RuntimeError(f"Isaac Sim evaluation did not write metrics JSON: {output_json_abs}")

    with open(output_json_abs, encoding="utf-8") as f:
        payload = json.load(f)
    return SimEvalResult.from_dict(payload)
