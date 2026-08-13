#!/usr/bin/env python3
"""Dependency-light geometry helpers for behavior surface interaction fields."""

from __future__ import annotations

import math

import numpy as np


def safe_unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    return value / np.float32(norm) if norm > 1e-8 else np.zeros_like(value)


def euler_xyz_rotation_matrix(euler: np.ndarray) -> np.ndarray:
    """Match PyBullet's roll-pitch-yaw matrix convention without PyBullet."""

    roll, pitch, yaw = (float(value) for value in euler)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def oriented_box_ray_distance(
    direction_world: np.ndarray,
    euler_world: np.ndarray,
    half_extents_local: np.ndarray,
) -> float:
    """Distance from box center to its surface along a world-frame ray."""

    direction = safe_unit(direction_world)
    rotation = euler_xyz_rotation_matrix(euler_world)
    direction_local = rotation.T @ direction
    active = np.abs(direction_local) > 1e-7
    if not np.any(active):
        return 0.0
    distances = np.asarray(half_extents_local, dtype=np.float32)[active] / np.abs(
        direction_local[active]
    )
    return float(np.min(distances))
