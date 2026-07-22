# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for lenient-teacher mask+depth data collection."""

from isaaclab.utils import configclass

from .vision_teacher_student_env_cfg import FrankaCubeLiftVisionMaskDepthPlayEnvCfg


@configclass
class FrankaCubeLiftLenientTeacherMaskDepthDataEnvCfg(FrankaCubeLiftVisionMaskDepthPlayEnvCfg):
    """Vision mask+depth env for collecting lenient-teacher demonstrations."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 30.0
