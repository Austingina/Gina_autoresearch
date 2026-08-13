#!/usr/bin/env python3
"""Small geometry regression test for the behavior block-surface G builder."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, Path(__file__).resolve().parent.as_posix())

from behavior_surface_geometry import oriented_box_ray_distance  # noqa: E402


def main() -> None:
    half_extents = np.asarray([0.02, 0.03, 0.04], dtype=np.float32)
    flow = np.asarray([-1.0, 0.0, 0.0], dtype=np.float32)
    distance_yaw0 = oriented_box_ray_distance(
        flow, np.asarray([0.0, 0.0, 0.0], dtype=np.float32), half_extents
    )
    distance_yaw90 = oriented_box_ray_distance(
        flow, np.asarray([0.0, 0.0, np.pi / 2], dtype=np.float32), half_extents
    )
    assert np.isclose(distance_yaw0, 0.02), distance_yaw0
    assert np.isclose(distance_yaw90, 0.03), distance_yaw90

    print("behavior block-surface G: OK")


if __name__ == "__main__":
    main()
