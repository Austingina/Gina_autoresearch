#!/usr/bin/env python3
"""Capture deterministic CALVIN camera observations for renderer parity checks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calvin-code", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sequence-index", type=int, default=87)
    args = parser.parse_args()

    calvin_code = args.calvin_code.resolve()
    out_dir = args.out_dir.resolve()
    calvin_models = calvin_code / "calvin_models"
    calvin_env = calvin_code / "calvin_env"
    calvin_agent = calvin_models / "calvin_agent"
    sys.path.insert(0, calvin_models.as_posix())
    sys.path.insert(0, calvin_env.as_posix())
    os.chdir(calvin_agent)

    from calvin_agent.evaluation.evaluate_policy import (  # noqa: PLC0415
        get_env_state_for_initial_condition,
    )
    from calvin_agent.evaluation.multistep_sequences import get_sequences  # noqa: PLC0415
    from calvin_env.envs.play_table_env import get_env  # noqa: PLC0415

    sequences = get_sequences(args.sequence_index + 1)
    initial_state, tasks = sequences[args.sequence_index]
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env = get_env(
        args.data_root.resolve() / "validation",
        show_gui=False,
        obs_space={
            "rgb_obs": ["rgb_static", "rgb_gripper"],
            "depth_obs": [],
        },
    )
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    obs = env.get_obs()

    out_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for camera in ("rgb_static", "rgb_gripper"):
        image = np.asarray(obs["rgb_obs"][camera], dtype=np.uint8)
        arrays[camera] = image
        Image.fromarray(image).save(out_dir / f"{camera}.png")
    np.savez_compressed(out_dir / "observations.npz", **arrays)
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "sequence_index": args.sequence_index,
                "tasks": list(tasks),
                "egl_visible_devices": os.environ.get("EGL_VISIBLE_DEVICES"),
            },
            f,
            indent=2,
        )

    if hasattr(env, "close"):
        env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
