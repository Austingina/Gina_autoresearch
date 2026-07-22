# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vision observations: GT instance mask (cube=white) + depth from dual cameras."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


CUBE_CLASS = "cube"
ROBOT_CLASS = "robot"
TABLE_CLASS = "table"
GROUND_CLASS = "ground"

CUBE_MASK_COLOR = (1.0, 1.0, 1.0)
FIXED_NON_CUBE_MASK_COLORS = (
    (0.10, 0.35, 0.95),  # robot / arm: blue
    (0.55, 0.35, 0.18),  # table: brown
    (0.10, 0.10, 0.10),  # ground / background: dark gray
)

NON_CUBE_CLASSES: tuple[tuple[str, int], ...] = (
    (ROBOT_CLASS, 0),
    (TABLE_CLASS, 1),
    (GROUND_CLASS, 2),
)


def _classify_prim_path(prim_path: str) -> str | None:
    """Map a USD prim path to a coarse object class used for mask coloring."""
    if "/Object" in prim_path:
        return CUBE_CLASS
    if "/Robot" in prim_path:
        return ROBOT_CLASS
    if "/Table" in prim_path:
        return TABLE_CLASS
    if "GroundPlane" in prim_path:
        return GROUND_CLASS
    return None


def _get_segmentation_info(env: ManagerBasedEnv, sensor_name: str, data_type: str) -> dict[str, Any]:
    """Read Replicator info dict for a camera segmentation annotator."""
    sensor = env.scene.sensors[sensor_name]
    info_container = sensor.data.info
    if info_container is None:
        return {}

    if isinstance(info_container, dict):
        raw = info_container.get(data_type)
        return raw if isinstance(raw, dict) else {}

    if isinstance(info_container, list):
        for entry in info_container:
            if not isinstance(entry, dict):
                continue
            raw = entry.get(data_type)
            if isinstance(raw, dict) and raw.get("idToLabels"):
                return raw
        if info_container and isinstance(info_container[0], dict):
            raw = info_container[0].get(data_type)
            return raw if isinstance(raw, dict) else {}

    return {}


def _build_instance_id_groups(info: dict[str, Any]) -> dict[str, list[int]]:
    """Group Replicator instance IDs by coarse object class using prim paths."""
    groups = {CUBE_CLASS: [], ROBOT_CLASS: [], TABLE_CLASS: [], GROUND_CLASS: []}
    for instance_id, prim_path in info.get("idToLabels", {}).items():
        class_name = _classify_prim_path(str(prim_path))
        if class_name is not None:
            groups[class_name].append(int(instance_id))
    return groups


def _get_instance_id_groups(env: ManagerBasedEnv, sensor_name: str) -> dict[str, list[int]]:
    cache_attr = f"_vision_instance_id_groups_{sensor_name}"
    if hasattr(env, cache_attr):
        return getattr(env, cache_attr)

    info = _get_segmentation_info(env, sensor_name, "instance_id_segmentation_fast")
    groups = _build_instance_id_groups(info)
    if groups[CUBE_CLASS]:
        setattr(env, cache_attr, groups)
    return groups


def _sensor_image_shape(env: ManagerBasedEnv, sensor) -> tuple[int, int]:
    # Prefer cfg dims: reading sensor.data before the first render can yield empty annotators.
    if sensor.cfg.height > 0 and sensor.cfg.width > 0:
        return int(sensor.cfg.height), int(sensor.cfg.width)
    if sensor.data.image_shape is not None:
        return int(sensor.data.image_shape[0]), int(sensor.data.image_shape[1])
    raise RuntimeError(f"Camera sensor {sensor.cfg.name} has no configured image shape.")


def _safe_camera_output(sensor) -> dict | None:
    """Read camera annotator output, returning None when buffers are not ready yet."""
    try:
        output = sensor.data.output
    except RuntimeError:
        return None
    return output if output else None


def _read_instance_id_map(env: ManagerBasedEnv, sensor) -> torch.Tensor | None:
    output = _safe_camera_output(sensor)
    if not output or "instance_id_segmentation_fast" not in output:
        return None
    tensor = output["instance_id_segmentation_fast"]
    if tensor is None:
        return None
    if tensor.shape[-1] != 1:
        raise ValueError(
            "Expected raw instance IDs with shape (N, H, W, 1). "
            "Set colorize_instance_id_segmentation=False on the camera cfg."
        )
    return tensor.squeeze(-1).to(torch.long)


def randomize_colored_mask_palette(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
) -> None:
    """Sample saturated non-white RGB for robot / table / ground."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device)

    num_envs = env.num_envs
    palette = torch.zeros(num_envs, 3, 3, device=env.device)
    for class_idx in range(3):
        colors = 0.20 + 0.70 * torch.rand((env_ids.shape[0], 3), device=env.device)
        near_white = colors.min(dim=-1).values > 0.82
        colors[near_white] *= 0.55
        palette[env_ids, class_idx] = colors
    if not hasattr(env, "_vision_mask_palette"):
        env._vision_mask_palette = torch.zeros(num_envs, 3, 3, device=env.device)
    env._vision_mask_palette[env_ids] = palette[env_ids]


def set_fixed_colored_mask_palette(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    robot_color: tuple[float, float, float] = FIXED_NON_CUBE_MASK_COLORS[0],
    table_color: tuple[float, float, float] = FIXED_NON_CUBE_MASK_COLORS[1],
    ground_color: tuple[float, float, float] = FIXED_NON_CUBE_MASK_COLORS[2],
) -> None:
    """Use fixed RGB colors for robot / table / ground mask classes."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device)

    colors = torch.tensor((robot_color, table_color, ground_color), dtype=torch.float32, device=env.device)
    colors = torch.clamp(colors, 0.0, 1.0)
    if not hasattr(env, "_vision_mask_palette"):
        env._vision_mask_palette = torch.zeros(env.num_envs, 3, 3, device=env.device)
    env._vision_mask_palette[env_ids] = colors.view(1, 3, 3).expand(env_ids.shape[0], 3, 3)


def _read_palette(env: ManagerBasedEnv) -> torch.Tensor:
    if not hasattr(env, "_vision_mask_palette"):
        randomize_colored_mask_palette(env, torch.arange(env.num_envs, device=env.device))
    return env._vision_mask_palette


def camera_colored_semantic_mask(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_camera"),
) -> torch.Tensor:
    """Return (num_envs, H, W, 3) colored GT mask; cube pixels are always white.

    Uses Replicator ``instance_id_segmentation_fast`` + prim-path grouping:
    - ``/Object`` -> white
    - ``/Robot``, ``/Table``, ``GroundPlane`` -> random non-white palette colors
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    height, width = _sensor_image_shape(env, sensor)
    instance_ids = _read_instance_id_map(env, sensor)
    if instance_ids is None:
        return torch.zeros(env.num_envs, height, width, 3, device=env.device)

    id_groups = _get_instance_id_groups(env, sensor_cfg.name)
    palette = _read_palette(env)
    num_envs = instance_ids.shape[0]

    mask_rgb = torch.zeros(num_envs, height, width, 3, device=env.device)
    for class_name, palette_idx in NON_CUBE_CLASSES:
        for instance_id in id_groups.get(class_name, []):
            class_mask = instance_ids == instance_id
            color = palette[:, palette_idx].view(num_envs, 1, 1, 3).expand(num_envs, height, width, 3)
            mask_rgb = torch.where(class_mask.unsqueeze(-1), color, mask_rgb)

    for instance_id in id_groups.get(CUBE_CLASS, []):
        cube_mask = instance_ids == instance_id
        mask_rgb = torch.where(
            cube_mask.unsqueeze(-1),
            mask_rgb.new_tensor(CUBE_MASK_COLOR),
            mask_rgb,
        )

    return mask_rgb


def camera_depth(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_camera"),
    max_depth: float = 2.0,
) -> torch.Tensor:
    """Return normalized depth in [0, 1] from ``distance_to_image_plane``."""
    sensor = env.scene.sensors[sensor_cfg.name]
    height, width = _sensor_image_shape(env, sensor)
    output = _safe_camera_output(sensor)
    if not output or "distance_to_image_plane" not in output:
        return torch.zeros(env.num_envs, height, width, 1, device=env.device)

    depth = output["distance_to_image_plane"].squeeze(-1).clone()
    depth[depth == float("inf")] = max_depth
    depth = torch.clamp(depth / max_depth, 0.0, 1.0)
    return depth.unsqueeze(-1)
