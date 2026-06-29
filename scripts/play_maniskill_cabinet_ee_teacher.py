"""Control-action teacher for ManiSkill OpenCabinetDrawer/Door.

This script uses only robot control actions (`env.step(action)`) to solve the
cabinet tasks. It does not set cabinet/door/drawer state.

Controller used:
  Fetch `pd_ee_delta_pos`
    arm     : 3D EE delta position (IK)
    gripper : open/close
    body    : head pan, head tilt, torso lift deltas
    base    : forward velocity, yaw velocity

The teacher is intentionally simple and debuggable:
  1. drive the mobile base in front of the handle
  2. move EE to a pre-contact pose
  3. close gripper / engage handle
  4. move EE along κ (prismatic axis or hinge tangent)

It records an RGB/depth sensor video and prints task metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import mani_skill  # noqa: F401

from goals.maniskill_bridge import env_kappa, extract_interaction_state, oracle_actuate_field


SUPPORTED_ENVS = ("OpenCabinetDrawer-v1", "OpenCabinetDoor-v1")


def _np(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _wrap_pi(x: float) -> float:
    return (x + np.pi) % (2 * np.pi) - np.pi


def _unit(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        if fallback is None:
            return np.zeros(3, dtype=np.float32)
        return np.asarray(fallback, dtype=np.float32).reshape(3)
    return (v / n).astype(np.float32)


def _depth_vis(depth: np.ndarray) -> np.ndarray:
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    valid = depth > 0
    out = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [2, 98])
        if hi <= lo:
            hi = lo + 1
        gray = (np.clip((depth.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        out = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    return out


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _project_point(obs: dict, cam: str, point_world: np.ndarray) -> tuple[int, int] | None:
    params = obs.get("sensor_param", {}).get(cam)
    if params is None:
        return None
    ext = _np(params["extrinsic_cv"])[0]  # world -> camera, OpenCV convention
    k = _np(params["intrinsic_cv"])[0]
    pw = np.asarray([point_world[0], point_world[1], point_world[2], 1.0], dtype=np.float32)
    pc = ext @ pw
    if pc[2] <= 1e-5:
        return None
    uvw = k @ pc[:3]
    u = int(round(float(uvw[0] / uvw[2])))
    v = int(round(float(uvw[1] / uvw[2])))
    return u, v


def _draw_marker(img: np.ndarray, uv: tuple[int, int] | None, color: tuple[int, int, int], label: str) -> None:
    if uv is None:
        return
    u, v = uv
    h, w = img.shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        return
    cv2.drawMarker(img, (u, v), color, markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
    cv2.putText(img, label, (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def _panel(
    obs: dict,
    env_id: str,
    step: int,
    phase: str,
    target_dist: float,
    tcp_handle_dist: float,
    open_frac: float,
    success: bool,
    handle_pos: np.ndarray,
    tcp_pos: np.ndarray,
) -> np.ndarray:
    rows = []
    for cam in obs["sensor_data"]:
        data = obs["sensor_data"][cam]
        rgb = _np(data["rgb"])[0]
        _draw_marker(rgb, _project_point(obs, cam, handle_pos), (0, 255, 0), "handle")
        _draw_marker(rgb, _project_point(obs, cam, tcp_pos), (255, 0, 0), "tcp")
        depth = _depth_vis(_np(data["depth"])[0])
        rows.append(np.concatenate([_label(rgb, f"{cam} rgb"), _label(depth, f"{cam} depth")], axis=1))
    img = np.concatenate(rows, axis=0)
    title = (
        f"{env_id} step={step} phase={phase} target_dist={target_dist:.3f} "
        f"tcp_handle={tcp_handle_dist:.3f} open={open_frac:.2f} success={success}"
    )
    cv2.putText(img, title, (6, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
    return img


@dataclass
class TeacherCfg:
    max_steps: int = 240
    fps: int = 20
    base_standoff: float = 0.68
    pre_offset: float = 0.18
    contact_offset: float = 0.025
    engage_offset: float = 0.01
    pull_distance: float = 0.55
    direction_sign: float = 1.0
    use_pose_control: bool = False
    direction_mode: str = "kappa"
    base_push: bool = False
    reach_thresh: float = 0.045
    seed: int = 0


class CabinetEETeacher:
    def __init__(self, env, env_id: str, cfg: TeacherCfg):
        self.env = env
        self.base = env.unwrapped
        self.env_id = env_id
        self.cfg = cfg
        self.kappa = env_kappa(env_id)
        self.action = np.zeros(env.action_space.shape, dtype=np.float32)
        self.mapping = self.base.agent.controller.action_mapping

    def _set(self, key: str, value: np.ndarray) -> None:
        s, e = self.mapping[key]
        self.action[s:e] = np.asarray(value, dtype=np.float32).reshape(-1)

    def _desired_gripper_quat_world(self, open_dir: np.ndarray) -> np.ndarray:
        approaching = _unit(-open_dir, fallback=np.array([1, 0, 0], dtype=np.float32))
        closing = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(approaching, closing))) > 0.9:
            closing = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        closing = _unit(closing - approaching * float(np.dot(approaching, closing)))
        ortho = _unit(np.cross(closing, approaching))
        mat = np.stack([ortho, closing, approaching], axis=1)
        # scipy uses xyzw, SAPIEN/ManiSkill uses wxyz
        q_xyzw = R.from_matrix(mat).as_quat()
        return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

    def _root_frame_delta_action(
        self,
        desired_world: np.ndarray,
        desired_quat_world: np.ndarray | None = None,
        gain: float = 1.8,
    ) -> tuple[np.ndarray, float]:
        tcp = _np(self.base.agent.tcp.pose.p)[0]
        delta_w = desired_world - tcp
        dist = float(np.linalg.norm(delta_w))
        root_T = _np(self.base.agent.torso_lift_link.pose.to_transformation_matrix())[0]
        delta_root = root_T[:3, :3].T @ delta_w
        # PDEE delta pos action is normalized to [-1, 1] then scaled to +/-0.1m.
        pos_action = np.clip((gain * delta_root / 0.1), -1.0, 1.0).astype(np.float32)
        if not self.cfg.use_pose_control:
            return pos_action, dist

        root_q = _np(self.base.agent.torso_lift_link.pose.q)[0]  # wxyz
        cur_q_base = _np(self.base.agent.controller.controllers["arm"].ee_pose_at_base.q)[0]
        if desired_quat_world is None:
            rot_action = np.zeros(3, dtype=np.float32)
        else:
            root_inv = R.from_quat([-root_q[1], -root_q[2], -root_q[3], root_q[0]])
            des_world = R.from_quat([desired_quat_world[1], desired_quat_world[2], desired_quat_world[3], desired_quat_world[0]])
            des_base = root_inv * des_world
            cur_base = R.from_quat([cur_q_base[1], cur_q_base[2], cur_q_base[3], cur_q_base[0]])
            delta = des_base * cur_base.inv()
            rotvec = delta.as_rotvec()
            # PDEEPoseController multiplies normalized rot action by rot_lower=-0.1.
            rot_action = np.clip(-rotvec / 0.1, -1.0, 1.0).astype(np.float32)
        arm = np.concatenate([pos_action, rot_action], axis=0).astype(np.float32)
        return arm, dist

    def _base_action(self, desired_base_xy: np.ndarray, handle_xy: np.ndarray) -> np.ndarray:
        qpos = _np(self.base.agent.robot.get_qpos())[0]
        base_xy = qpos[:2]
        theta = float(qpos[2])
        to_goal = desired_base_xy - base_xy
        desired_theta = float(np.arctan2(handle_xy[1] - base_xy[1], handle_xy[0] - base_xy[0]))
        yaw_err = _wrap_pi(desired_theta - theta)
        forward_axis = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        forward_err = float(np.dot(to_goal, forward_axis))
        forward = np.clip(1.8 * forward_err, -0.9, 0.9)
        yaw = np.clip(1.8 * yaw_err / np.pi, -0.9, 0.9)
        # Freeze forward if facing away; rotate first.
        if abs(yaw_err) > 0.45:
            forward = 0.0
        return np.array([forward, yaw], dtype=np.float32)

    def act(self, step: int) -> tuple[np.ndarray, str, float, float, bool]:
        state = extract_interaction_state(self.env, self.env_id)
        handle = state["handle_pos"][0]
        open_dir = oracle_actuate_field(state, self.kappa)[0]
        open_dir = _unit(open_dir, fallback=np.array([-1, 0, 0], dtype=np.float32))
        if self.cfg.direction_mode == "radial" and "hinge_origin" in state:
            open_dir = _unit(state["handle_pos"][0] - state["hinge_origin"][0], fallback=open_dir)
        elif self.cfg.direction_mode == "x":
            open_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif self.cfg.direction_mode == "y":
            open_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        open_dir = open_dir * float(self.cfg.direction_sign)
        handle_xy = handle[:2]
        desired_base_xy = handle_xy - open_dir[:2] * self.cfg.base_standoff

        # Phase schedule. Dynamic dist can advance early; time schedule avoids deadlock.
        if step < 65:
            phase = "base_align"
            ee_target = handle - open_dir * self.cfg.pre_offset + np.array([0, 0, 0.02], dtype=np.float32)
            gripper = 1.0
        elif step < 120:
            phase = "reach_precontact"
            ee_target = handle - open_dir * self.cfg.contact_offset
            gripper = 1.0
        elif step < 150:
            phase = "close_engage"
            ee_target = handle + open_dir * self.cfg.engage_offset
            gripper = -1.0
        else:
            phase = "actuate"
            # Door tangent changes with current handle pose; recompute every step.
            alpha = min(1.0, (step - 150) / max(self.cfg.max_steps - 150, 1))
            ee_target = handle + open_dir * (self.cfg.engage_offset + self.cfg.pull_distance * alpha)
            gripper = -1.0

        desired_q = self._desired_gripper_quat_world(open_dir) if self.cfg.use_pose_control else None
        arm, dist = self._root_frame_delta_action(ee_target, desired_q)
        base_cmd = self._base_action(desired_base_xy, handle_xy)
        if phase == "actuate" and self.cfg.base_push and np.linalg.norm(open_dir[:2]) > 1e-4:
            qpos = _np(self.base.agent.robot.get_qpos())[0]
            theta = float(qpos[2])
            desired_theta = float(np.arctan2(open_dir[1], open_dir[0]))
            yaw_err = _wrap_pi(desired_theta - theta)
            forward = 0.55 if abs(yaw_err) < 0.55 else 0.0
            yaw = np.clip(2.0 * yaw_err / np.pi, -1.0, 1.0)
            base_cmd = np.array([forward, yaw], dtype=np.float32)

        # Torso tracks handle z roughly; head stays neutral.
        tcp_z = float(_np(self.base.agent.tcp.pose.p)[0, 2])
        torso_cmd = np.clip((handle[2] - tcp_z) / 0.12, -1.0, 1.0)

        self.action.fill(0.0)
        self._set("arm", arm)
        self._set("gripper", np.array([gripper], dtype=np.float32))
        self._set("body", np.array([0.0, 0.0, torso_cmd], dtype=np.float32))
        self._set("base", base_cmd)

        qpos = _np(self.base.handle_link.joint.qpos)[0]
        lim = _np(self.base.handle_link.joint.limits)[0]
        open_frac = float((qpos - lim[0]) / max(lim[1] - lim[0], 1e-6))
        success = bool(_np(self.base.evaluate()["success"])[0])
        return self.action.copy(), phase, dist, open_frac, success


def run(env_id: str, out_dir: Path, cfg: TeacherCfg) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode="rgb+depth",
        control_mode="pd_ee_delta_pose" if cfg.use_pose_control else "pd_ee_delta_pos",
        sim_backend="physx_cuda",
    )
    obs, _ = env.reset(seed=cfg.seed)
    teacher = CabinetEETeacher(env, env_id, cfg)
    frames = []
    success = False
    best_open = 0.0
    log_rows = ["step,phase,target_dist,tcp_handle_dist,open_frac,success,contact_force"]

    for step in range(cfg.max_steps):
        action, phase, dist, open_frac, success = teacher.act(step)
        obs, _, terminated, truncated, info = env.step(action)
        success = bool(_np(info["success"])[0])
        f1 = env.unwrapped.scene.get_pairwise_contact_forces(
            env.unwrapped.agent.finger1_link, env.unwrapped.handle_link
        )
        f2 = env.unwrapped.scene.get_pairwise_contact_forces(
            env.unwrapped.agent.finger2_link, env.unwrapped.handle_link
        )
        contact_force = float(torch.linalg.norm(f1, axis=1)[0] + torch.linalg.norm(f2, axis=1)[0])
        state = extract_interaction_state(env, env_id)
        handle_pos = state["handle_pos"][0]
        tcp_pos = _np(env.unwrapped.agent.tcp.pose.p)[0]
        tcp_handle_dist = float(np.linalg.norm(tcp_pos - handle_pos))
        best_open = max(best_open, open_frac)
        log_rows.append(
            f"{step},{phase},{dist:.5f},{tcp_handle_dist:.5f},{open_frac:.5f},{int(success)},{contact_force:.5f}"
        )
        if step % 2 == 0:
            frames.append(
                _panel(
                    obs,
                    env_id,
                    step,
                    phase,
                    dist,
                    tcp_handle_dist,
                    open_frac,
                    success,
                    handle_pos,
                    tcp_pos,
                )
            )
        if success and step > 170:
            break

    video_path = out_dir / f"{env_id}_ee_teacher.mp4"
    imageio.mimsave(video_path, frames, fps=cfg.fps)
    (out_dir / f"{env_id}_ee_teacher_log.csv").write_text("\n".join(log_rows) + "\n")
    env.close()
    print(f"env_id={env_id}")
    print(f"success={success}")
    print(f"best_open_frac={best_open:.3f}")
    print(f"video={video_path}")
    print(f"log={out_dir / f'{env_id}_ee_teacher_log.csv'}")
    return success


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", choices=SUPPORTED_ENVS, required=True)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--max_steps", type=int, default=240)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pull_distance", type=float, default=0.55)
    parser.add_argument("--engage_offset", type=float, default=0.01)
    parser.add_argument("--direction_sign", type=float, default=1.0)
    parser.add_argument("--direction_mode", choices=["kappa", "radial", "x", "y"], default="kappa")
    parser.add_argument("--base_push", action="store_true")
    parser.add_argument("--pose_control", action="store_true")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path("runs") / f"maniskill_{args.env_id}_ee_teacher_{ts}"
    cfg = TeacherCfg(
        max_steps=args.max_steps,
        fps=args.fps,
        seed=args.seed,
        pull_distance=args.pull_distance,
        engage_offset=args.engage_offset,
        direction_sign=args.direction_sign,
        use_pose_control=args.pose_control,
        direction_mode=args.direction_mode,
        base_push=args.base_push,
    )
    ok = run(args.env_id, out_dir, cfg)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
