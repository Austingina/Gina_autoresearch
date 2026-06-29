"""Load mask trajectories and build guidance-field supervision targets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from goals.interaction_geometry import DEFAULT_LIFT_HEIGHT_ABOVE_CUBE_M, subgoal_direction_2d_xz

MASK_KEYS = ("ee_mask", "third_person_mask")
MASK_DEPTH_KEYS = ("ee_mask", "ee_depth", "third_person_mask", "third_person_depth")
FIELD_VISUAL_CHANNELS_MASK = 6
FIELD_VISUAL_CHANNELS_MASK_DEPTH = 8


def _demo_sort_key(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else 0


def _read_dataset(group: h5py.Group, path: str) -> np.ndarray:
    node = group
    for part in path.split("/"):
        node = node[part]
    return np.asarray(node)


def mask_centroid_norm(mask_hwc: np.ndarray, *, threshold: float = 0.25) -> np.ndarray | None:
    """Return (cx, cy) in [0, 1] image coordinates from colored mask (H, W, C)."""
    if mask_hwc.ndim == 2:
        m = mask_hwc
    else:
        m = mask_hwc.max(axis=-1)
    h, w = m.shape
    ys, xs = np.where(m > threshold)
    if len(xs) < 5:
        return None
    return np.array([xs.mean() / max(w - 1, 1), ys.mean() / max(h - 1, 1)], dtype=np.float32)


def field_target_from_centroids(c0: np.ndarray, c1: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    delta = c1 - c0
    norm = float(np.linalg.norm(delta))
    if norm < eps:
        return np.zeros(2, dtype=np.float32)
    return (delta / norm).astype(np.float32)


def _to_chw(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if x.ndim == 3 and x.shape[-1] in (1, 3):
        return x.transpose(2, 0, 1)
    if x.ndim == 2:
        return x[np.newaxis, ...]
    return x


def stack_mask_pair(ee_mask: np.ndarray, tp_mask: np.ndarray) -> np.ndarray:
    """Return float32 (6, H, W) from (H,W,3) masks."""
    return np.concatenate([_to_chw(ee_mask), _to_chw(tp_mask)], axis=0)


def stack_maskdepth_visual(
    ee_mask: np.ndarray,
    ee_depth: np.ndarray,
    tp_mask: np.ndarray,
    tp_depth: np.ndarray,
) -> np.ndarray:
    """Return float32 (8, H, W): ee_mask(3)+ee_depth(1)+tp_mask(3)+tp_depth(1)."""
    if ee_depth.ndim == 3 and ee_depth.shape[-1] == 1:
        ee_depth = ee_depth[..., 0]
    if tp_depth.ndim == 3 and tp_depth.shape[-1] == 1:
        tp_depth = tp_depth[..., 0]
    return np.concatenate(
        [_to_chw(ee_mask), _to_chw(ee_depth), _to_chw(tp_mask), _to_chw(tp_depth)],
        axis=0,
    )


def resize_visual_chw(visual: np.ndarray, image_size: int) -> np.ndarray:
    if not image_size or visual.shape[-1] == image_size:
        return visual.astype(np.float32)
    import cv2

    c, _, _ = visual.shape
    resized = np.empty((c, image_size, image_size), dtype=np.float32)
    for i in range(c):
        resized[i] = cv2.resize(visual[i], (image_size, image_size), interpolation=cv2.INTER_AREA)
    return resized


@dataclass
class FieldSample:
    visual: np.ndarray  # (C, H, W) — C=6 mask-only or 8 mask+depth
    target: np.ndarray  # (2,)
    phase: float
    demo: str
    step: int

    @property
    def mask(self) -> np.ndarray:
        """Backward-compatible alias."""
        return self.visual


def _state_goal_target(student_state: np.ndarray, t: int) -> np.ndarray | None:
    """2D relocation direction in robot x-z plane from cube toward goal."""
    cube = student_state[t, 4:7]
    goal = student_state[t, 7:10]
    vec = np.array([goal[0] - cube[0], goal[2] - cube[2]], dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-4:
        return None
    return (vec / norm).astype(np.float32)


def _subgoal_target(
    student_state: np.ndarray,
    t: int,
    phase: float,
    *,
    lift_height_above: float = DEFAULT_LIFT_HEIGHT_ABOVE_CUBE_M,
    out_dim: int = 2,
    ee_pose: np.ndarray | None = None,
) -> np.ndarray | None:
    """Phase-aware lift G* (ψ): REACH EE→cube; ACTUATE opposite EE forward axis."""
    if out_dim == 3:
        from goals.interaction_geometry import subgoal_direction_3d

        return subgoal_direction_3d(student_state, t, phase, height_above=lift_height_above, ee_pose=ee_pose)
    return subgoal_direction_2d_xz(student_state, t, phase)


def load_field_samples_from_hdf5(
    path: Path,
    *,
    mask_key: str = "third_person_mask",
    success_only: bool = True,
    post_grasp_only: bool = True,
    image_size: int = 64,
    supervision: str = "state_goal",
    input_mode: str = "mask_depth",
    out_dim: int = 2,
) -> list[FieldSample]:
    if input_mode not in ("mask", "mask_depth"):
        raise ValueError(f"Unknown input_mode={input_mode!r}")
    use_depth = input_mode == "mask_depth"
    samples: list[FieldSample] = []
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys(), key=_demo_sort_key)
        for demo_name in demos:
            g = f["data"][demo_name]
            if success_only and not bool(g.attrs.get("success", False)):
                continue
            ee_mask = _read_dataset(g, "obs/ee_mask")
            tp_mask = _read_dataset(g, "obs/third_person_mask")
            ee_depth = _read_dataset(g, "obs/ee_depth") if use_depth else None
            tp_depth = _read_dataset(g, "obs/third_person_depth") if use_depth else None
            student_state = _read_dataset(g, "obs/student_state").astype(np.float32)
            ee_pose = _read_dataset(g, "ee_pose").astype(np.float32) if "ee_pose" in g else None
            phase = _read_dataset(g, "goals/phase").reshape(-1)
            lengths = [len(ee_mask), len(tp_mask), len(phase), len(student_state)]
            if use_depth:
                lengths.extend([len(ee_depth), len(tp_depth)])
            t_steps = min(lengths) - 1
            for t in range(t_steps):
                ph = float(phase[t])
                if post_grasp_only and ph < 0.5:
                    continue
                if supervision == "mask_flow":
                    c0 = mask_centroid_norm(tp_mask[t] if mask_key == "third_person_mask" else ee_mask[t])
                    c1 = mask_centroid_norm(tp_mask[t + 1] if mask_key == "third_person_mask" else ee_mask[t + 1])
                    if c0 is None or c1 is None:
                        continue
                    target = field_target_from_centroids(c0, c1)
                elif supervision == "state_goal":
                    target = _state_goal_target(student_state, t)
                elif supervision == "subgoal":
                    target = _subgoal_target(student_state, t, ph, out_dim=out_dim, ee_pose=ee_pose)
                else:
                    raise ValueError(f"Unknown supervision={supervision!r}")
                if target is None or float(np.linalg.norm(target)) < 1e-5:
                    continue
                if use_depth:
                    visual = stack_maskdepth_visual(ee_mask[t], ee_depth[t], tp_mask[t], tp_depth[t])
                else:
                    visual = stack_mask_pair(ee_mask[t], tp_mask[t])
                visual = resize_visual_chw(visual, image_size)
                samples.append(
                    FieldSample(visual=visual.astype(np.float32), target=target, phase=ph, demo=demo_name, step=t)
                )
    return samples


def load_demo_dir(
    demo_dir: Path,
    *,
    image_size: int = 64,
    success_only: bool = True,
    post_grasp_only: bool = True,
    supervision: str = "state_goal",
    input_mode: str = "mask_depth",
    out_dim: int = 2,
) -> list[FieldSample]:
    files = sorted(demo_dir.glob("*.hdf5"))
    out: list[FieldSample] = []
    for fp in files:
        out.extend(
            load_field_samples_from_hdf5(
                fp,
                image_size=image_size,
                success_only=success_only,
                post_grasp_only=post_grasp_only,
                supervision=supervision,
                input_mode=input_mode,
                out_dim=out_dim,
            )
        )
    return out


def train_val_split(samples: list[FieldSample], val_ratio: float = 0.2, seed: int = 42) -> tuple[list[FieldSample], list[FieldSample]]:
    demos = sorted({s.demo for s in samples})
    rng = np.random.default_rng(seed)
    rng.shuffle(demos)
    n_val = max(1, int(round(len(demos) * val_ratio)))
    val_set = set(demos[:n_val])
    train = [s for s in samples if s.demo not in val_set]
    val = [s for s in samples if s.demo in val_set]
    return train, val


@dataclass(frozen=True)
class FieldIndexEntry:
    """Lightweight index for lazy HDF5 loading (visual read per step, target precomputed)."""

    hdf5_path: str
    demo: str
    step: int
    target: np.ndarray  # (2,) or (3,)
    phase: float = 0.0  # ψ source: contact_latched = 1.0 if phase >= 0.5 else 0.0


def _resolve_field_target(
    supervision: str,
    *,
    student_state: np.ndarray,
    t: int,
    phase: float,
    ee_pose: np.ndarray | None = None,
    ee_mask_t: np.ndarray | None = None,
    ee_mask_t1: np.ndarray | None = None,
    tp_mask_t: np.ndarray | None = None,
    tp_mask_t1: np.ndarray | None = None,
    mask_key: str = "third_person_mask",
    out_dim: int = 2,
) -> np.ndarray | None:
    if supervision == "mask_flow":
        if tp_mask_t is None or tp_mask_t1 is None or ee_mask_t is None or ee_mask_t1 is None:
            raise ValueError("mask_flow requires mask frames at t and t+1")
        use_tp = mask_key == "third_person_mask"
        c0 = mask_centroid_norm(tp_mask_t if use_tp else ee_mask_t)
        c1 = mask_centroid_norm(tp_mask_t1 if use_tp else ee_mask_t1)
        if c0 is None or c1 is None:
            return None
        target = field_target_from_centroids(c0, c1)
    elif supervision == "state_goal":
        target = _state_goal_target(student_state, t)
    elif supervision == "subgoal":
        target = _subgoal_target(student_state, t, phase, out_dim=out_dim, ee_pose=ee_pose)
    else:
        raise ValueError(f"Unknown supervision={supervision!r}")
    if target is None or float(np.linalg.norm(target)) < 1e-5:
        return None
    return target.astype(np.float32)


def build_field_indices_from_hdf5(
    path: Path,
    *,
    mask_key: str = "third_person_mask",
    success_only: bool = True,
    post_grasp_only: bool = True,
    supervision: str = "state_goal",
    input_mode: str = "mask_depth",
    out_dim: int = 2,
) -> list[FieldIndexEntry]:
    """Build sample indices without loading full mask trajectories into RAM."""
    if input_mode not in ("mask", "mask_depth"):
        raise ValueError(f"Unknown input_mode={input_mode!r}")
    use_depth = input_mode == "mask_depth"
    indices: list[FieldIndexEntry] = []
    path_str = str(path.resolve())
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys(), key=_demo_sort_key)
        for demo_name in demos:
            g = f["data"][demo_name]
            if success_only and not bool(g.attrs.get("success", False)):
                continue
            t_steps = int(g["obs/ee_mask"].shape[0]) - 1
            student_state = np.asarray(g["obs/student_state"], dtype=np.float32)
            ee_pose = np.asarray(g["ee_pose"], dtype=np.float32) if "ee_pose" in g else None
            phase = np.asarray(g["goals/phase"], dtype=np.float32).reshape(-1)
            t_steps = min(t_steps, len(student_state) - 1, len(phase) - 1)
            if use_depth:
                t_steps = min(t_steps, int(g["obs/ee_depth"].shape[0]) - 1, int(g["obs/third_person_depth"].shape[0]) - 1)
            for t in range(t_steps):
                ph = float(phase[t])
                if post_grasp_only and ph < 0.5:
                    continue
                if supervision == "mask_flow":
                    ee_t = np.asarray(g["obs/ee_mask"][t])
                    ee_t1 = np.asarray(g["obs/ee_mask"][t + 1])
                    tp_t = np.asarray(g["obs/third_person_mask"][t])
                    tp_t1 = np.asarray(g["obs/third_person_mask"][t + 1])
                    target = _resolve_field_target(
                        supervision,
                        student_state=student_state,
                        t=t,
                        phase=ph,
                        ee_pose=ee_pose,
                        ee_mask_t=ee_t,
                        ee_mask_t1=ee_t1,
                        tp_mask_t=tp_t,
                        tp_mask_t1=tp_t1,
                        mask_key=mask_key,
                        out_dim=out_dim,
                    )
                else:
                    target = _resolve_field_target(
                        supervision, student_state=student_state, t=t, phase=ph, ee_pose=ee_pose, out_dim=out_dim
                    )
                if target is None:
                    continue
                indices.append(
                    FieldIndexEntry(hdf5_path=path_str, demo=demo_name, step=t, target=target, phase=ph)
                )
    return indices


def build_field_indices_from_demo_dir(
    demo_dir: Path,
    *,
    success_only: bool = True,
    post_grasp_only: bool = True,
    supervision: str = "state_goal",
    input_mode: str = "mask_depth",
    out_dim: int = 2,
) -> list[FieldIndexEntry]:
    out: list[FieldIndexEntry] = []
    for fp in sorted(demo_dir.glob("*.hdf5")):
        out.extend(
            build_field_indices_from_hdf5(
                fp,
                success_only=success_only,
                post_grasp_only=post_grasp_only,
                supervision=supervision,
                input_mode=input_mode,
                out_dim=out_dim,
            )
        )
    return out


def train_val_split_indices(
    indices: list[FieldIndexEntry], val_ratio: float = 0.2, seed: int = 42
) -> tuple[list[FieldIndexEntry], list[FieldIndexEntry]]:
    demos = sorted({e.demo for e in indices})
    rng = np.random.default_rng(seed)
    rng.shuffle(demos)
    n_val = max(1, int(round(len(demos) * val_ratio)))
    val_set = set(demos[:n_val])
    train = [e for e in indices if e.demo not in val_set]
    val = [e for e in indices if e.demo in val_set]
    return train, val


def read_field_visual_at_index(
    entry: FieldIndexEntry,
    *,
    image_size: int,
    input_mode: str = "mask_depth",
    files: dict[str, h5py.File] | None = None,
) -> np.ndarray:
    """Read and resize one (C,H,W) visual tensor for a lazy index entry."""
    use_depth = input_mode == "mask_depth"
    own_files = files is None
    if files is None:
        files = {}
    if entry.hdf5_path not in files:
        files[entry.hdf5_path] = h5py.File(entry.hdf5_path, "r")
    g = files[entry.hdf5_path]["data"][entry.demo]
    t = entry.step
    ee_mask = np.asarray(g["obs/ee_mask"][t])
    tp_mask = np.asarray(g["obs/third_person_mask"][t])
    if use_depth:
        ee_depth = np.asarray(g["obs/ee_depth"][t])
        tp_depth = np.asarray(g["obs/third_person_depth"][t])
        visual = stack_maskdepth_visual(ee_mask, ee_depth, tp_mask, tp_depth)
    else:
        visual = stack_mask_pair(ee_mask, tp_mask)
    return resize_visual_chw(visual, image_size)
