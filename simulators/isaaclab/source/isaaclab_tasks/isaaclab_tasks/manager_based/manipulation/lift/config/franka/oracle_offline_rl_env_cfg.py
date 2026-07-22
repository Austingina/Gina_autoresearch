# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for fixed-scenario offline RL data collection."""

from isaaclab.utils import configclass

from .oracle_data_collection_env_cfg import FrankaCubeLiftOracleDataCollectionEnvCfg


@configclass
class FrankaCubeLiftOracleOfflineRLEnvCfg(FrankaCubeLiftOracleDataCollectionEnvCfg):
    """Oracle data collection with a fixed cube spawn pose at the task default."""

    def __post_init__(self):
        super().__post_init__()

        # Keep the cube at the default centered spawn pose: [0.5, 0, 0.055].
        self.events.reset_object_position.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
        }
