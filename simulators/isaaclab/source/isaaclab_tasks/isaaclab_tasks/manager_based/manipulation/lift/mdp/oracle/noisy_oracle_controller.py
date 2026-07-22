# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Oracle controller with held random action noise for offline RL data collection."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils

from .lift_cube_oracle import LiftCubeOraclePolicy


class NoisyOracleActionController:
    """Mix oracle actions with held random sub-optimal actions.

    Each environment maintains a countdown flag. When a new noise segment starts, a random
    perturbed action is sampled and reused for a random number of steps in ``[hold_min, hold_max]``.
    """

    def __init__(
        self,
        oracle: LiftCubeOraclePolicy,
        *,
        noise_prob: float = 0.2,
        hold_min: int = 5,
        hold_max: int = 10,
        pos_noise_std: float = 0.03,
        rot_noise_std: float = 0.15,
        gripper_flip_prob: float = 0.3,
    ):
        if hold_min < 1 or hold_max < hold_min:
            raise ValueError(f"Invalid hold range: [{hold_min}, {hold_max}]")

        self.oracle = oracle
        self.device = oracle.device
        self.num_envs = oracle.num_envs
        self.noise_prob = noise_prob
        self.hold_min = hold_min
        self.hold_max = hold_max
        self.pos_noise_std = pos_noise_std
        self.rot_noise_std = rot_noise_std
        self.gripper_flip_prob = gripper_flip_prob

        self.noise_flag = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.noise_action = torch.zeros(self.num_envs, 8, device=self.device)
        self.last_use_noise = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def uses_ik_abs(self) -> bool:
        return self.oracle.uses_ik_abs

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        """Reset oracle and per-environment noise state."""
        self.oracle.reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self.noise_flag[env_ids] = 0

    def compute(self) -> torch.Tensor:
        """Compute actions with oracle/noise mixing."""
        self.noise_flag -= 1
        oracle_actions = self.oracle.compute()

        rand = torch.rand(self.num_envs, device=self.device)
        enter_noise = (rand < self.noise_prob) & (self.noise_flag <= 0)
        continue_noise = self.noise_flag > 0
        use_noise = enter_noise | continue_noise

        if enter_noise.any():
            sampled_actions = self._sample_noisy_action(oracle_actions)
            self.noise_action[enter_noise] = sampled_actions[enter_noise]
            num_new = int(enter_noise.sum().item())
            self.noise_flag[enter_noise] = torch.randint(
                self.hold_min,
                self.hold_max + 1,
                (num_new,),
                device=self.device,
                dtype=torch.int32,
            )

        actions = oracle_actions.clone()
        if use_noise.any():
            actions[use_noise] = self.noise_action[use_noise]
        self.last_use_noise = use_noise
        return actions

    def _sample_noisy_action(self, oracle_actions: torch.Tensor) -> torch.Tensor:
        """Sample a random sub-optimal action around the current oracle command."""
        noisy = oracle_actions.clone()
        noisy[:, :3] += torch.randn(self.num_envs, 3, device=self.device) * self.pos_noise_std

        rot_delta = torch.randn(self.num_envs, 3, device=self.device) * self.rot_noise_std
        noisy[:, 3:7] = math_utils.quat_unique(math_utils.quat_box_plus(noisy[:, 3:7], rot_delta))

        if self.gripper_flip_prob > 0.0:
            flip = torch.rand(self.num_envs, device=self.device) < self.gripper_flip_prob
            noisy[:, 7] = torch.where(flip, -noisy[:, 7], noisy[:, 7])

        return noisy
