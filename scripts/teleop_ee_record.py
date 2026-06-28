#!/usr/bin/env python3
"""Keyboard-control Franka EE and record a human demonstration trajectory."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from types import SimpleNamespace
from typing import Any

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Keyboard EE teleop recorder for Arena visual tasks.")
parser.add_argument("--arena_task", choices=["button", "door", "microwave"], default="button")
parser.add_argument("--output_dir", type=str, default="/home/l/code/AutoResearch/runs/ee_teleop")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_spacing", type=float, default=30.0)
parser.add_argument("--pos_step", type=float, default=0.030)
parser.add_argument("--rot_step", type=float, default=0.080)
parser.add_argument("--max_steps", type=int, default=100000)
parser.add_argument("--trail_interval", type=int, default=3)
parser.add_argument("--episode_length_s", type=float, default=3600.0)
parser.add_argument(
    "--allow_task_end",
    action="store_true",
    default=False,
    help="Allow normal success termination. By default teleop disables success termination while recording.",
)
parser.add_argument("--franka_x", type=float, default=None)
parser.add_argument("--franka_y", type=float, default=None)
parser.add_argument("--franka_z", type=float, default=None)
parser.add_argument("--franka_yaw_deg", type=float, default=None)
parser.add_argument("--platform_height", type=float, default=0.45)
parser.add_argument("--platform_size_x", type=float, default=0.80)
parser.add_argument("--platform_size_y", type=float, default=0.80)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder  # noqa: E402
from isaaclab_arena_environments.franka_open_fridge_environment import FrankaOpenFridgeEnvironment  # noqa: E402
from isaaclab_arena_environments.franka_open_microwave_environment import FrankaOpenMicrowaveEnvironment  # noqa: E402
from isaaclab_arena_environments.press_button_environment import PressButtonEnvironment  # noqa: E402


class KeyboardEEController:
    """Track pressed keys from the Isaac app window."""

    def __init__(self):
        self.keys: set[str] = set()
        self.quit = False
        self.mark = False
        self.reset_requested = False
        self._sub = None
        self._input = None
        self._keyboard = None
        self._setup()

    def _setup(self) -> None:
        import omni.appwindow

        app_window = omni.appwindow.get_default_app_window()
        self._keyboard = app_window.get_keyboard()
        self._input = carb.input.acquire_input_interface()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)

    def close(self) -> None:
        if self._sub is not None and self._input is not None and self._keyboard is not None:
            try:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            except Exception:
                pass

    def consume_mark(self) -> bool:
        value = self.mark
        self.mark = False
        return value

    def consume_reset(self) -> bool:
        value = self.reset_requested
        self.reset_requested = False
        return value

    def action(self, *, device: torch.device, action_shape: tuple[int, ...]) -> torch.Tensor:
        action = torch.zeros(action_shape, dtype=torch.float32, device=device)
        if action.ndim == 1:
            action = action.unsqueeze(0)

        # World-frame translation for the relative IK command.
        delta = torch.zeros((action.shape[0], 3), dtype=torch.float32, device=device)
        if "W" in self.keys:
            delta[:, 1] += args_cli.pos_step
        if "S" in self.keys:
            delta[:, 1] -= args_cli.pos_step
        if "D" in self.keys:
            delta[:, 0] += args_cli.pos_step
        if "A" in self.keys:
            delta[:, 0] -= args_cli.pos_step
        if "E" in self.keys:
            delta[:, 2] += args_cli.pos_step
        if "Q" in self.keys:
            delta[:, 2] -= args_cli.pos_step

        rot = torch.zeros((action.shape[0], 3), dtype=torch.float32, device=device)
        if "UP" in self.keys:
            rot[:, 0] += args_cli.rot_step
        if "DOWN" in self.keys:
            rot[:, 0] -= args_cli.rot_step
        if "LEFT" in self.keys:
            rot[:, 1] += args_cli.rot_step
        if "RIGHT" in self.keys:
            rot[:, 1] -= args_cli.rot_step
        if "Z" in self.keys:
            rot[:, 2] += args_cli.rot_step
        if "X" in self.keys:
            rot[:, 2] -= args_cli.rot_step

        gripper = 1.0
        if "C" in self.keys:
            gripper = -1.0
        if "V" in self.keys:
            gripper = 1.0

        action[:, :3] = delta
        if action.shape[-1] >= 7:
            action[:, 3:6] = rot
            action[:, 6] = gripper
        elif action.shape[-1] >= 4:
            action[:, -1] = gripper
        return action

    def _on_keyboard_event(self, event) -> bool:
        key = _key_name(event.input)
        if not key:
            return True
        pressed = event.type in (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_REPEAT)
        released = event.type == carb.input.KeyboardEventType.KEY_RELEASE
        if pressed:
            if key == "ESCAPE":
                self.quit = True
            elif key == "ENTER":
                self.mark = True
            elif key == "R":
                self.reset_requested = True
            else:
                self.keys.add(key)
        elif released:
            self.keys.discard(key)
        return True


def _key_name(input_key: Any) -> str:
    name = str(input_key).split(".")[-1].upper()
    aliases = {
        "ARROW_UP": "UP",
        "ARROW_DOWN": "DOWN",
        "ARROW_LEFT": "LEFT",
        "ARROW_RIGHT": "RIGHT",
        "NUMPAD_ENTER": "ENTER",
    }
    return aliases.get(name, name)


def _env_args() -> SimpleNamespace:
    common = dict(
        enable_cameras=True,
        embodiment="franka",
        teleop_device=None,
        object=None,
        num_envs=args_cli.num_envs,
        env_spacing=args_cli.env_spacing,
        mimic=False,
        solve_relations=True,
        placement_seed=None,
        disable_fabric=getattr(args_cli, "disable_fabric", False),
        device=getattr(args_cli, "device", "cuda:0"),
        seed=getattr(args_cli, "seed", None),
    )
    if args_cli.arena_task == "button":
        return SimpleNamespace(
            **common,
            kitchen_layout=1,
            kitchen_style=2,
            coffee_button_joint="Button002_joint",
            pressedness_threshold=0.5 if args_cli.allow_task_end else 1.1,
            reset_pressedness=0.0,
            episode_length_s=args_cli.episode_length_s,
            press_target_x=0.455,
            press_target_y=-0.331,
            press_target_z=1.117,
            franka_x=0.60 if args_cli.franka_x is None else args_cli.franka_x,
            franka_y=-0.72 if args_cli.franka_y is None else args_cli.franka_y,
            franka_z=args_cli.franka_z,
            franka_yaw_deg=90.0 if args_cli.franka_yaw_deg is None else args_cli.franka_yaw_deg,
            platform_height=args_cli.platform_height,
            platform_size_x=args_cli.platform_size_x,
            platform_size_y=args_cli.platform_size_y,
        )
    if args_cli.arena_task == "door":
        return SimpleNamespace(
            **common,
            kitchen_layout=1,
            kitchen_style=2,
            franka_x=4.45 if args_cli.franka_x is None else args_cli.franka_x,
            franka_y=-1.20 if args_cli.franka_y is None else args_cli.franka_y,
            franka_z=args_cli.franka_z,
            franka_yaw_deg=90.0 if args_cli.franka_yaw_deg is None else args_cli.franka_yaw_deg,
            reset_openness=0.0,
            openness_threshold=0.8 if args_cli.allow_task_end else 1.1,
            episode_length_s=args_cli.episode_length_s,
            open_target_x=4.25,
            open_target_y=-0.62,
            open_target_z=1.05,
            platform_height=args_cli.platform_height,
            platform_size_x=args_cli.platform_size_x,
            platform_size_y=args_cli.platform_size_y,
        )
    return SimpleNamespace(
        **common,
        reset_openness=0.0,
        openness_threshold=0.8 if args_cli.allow_task_end else 1.1,
        episode_length_s=args_cli.episode_length_s,
        open_target_x=0.42,
        open_target_y=-0.34,
        open_target_z=0.72,
    )


def _build_env(env_args: SimpleNamespace):
    env_cls = {
        "button": PressButtonEnvironment,
        "door": FrankaOpenFridgeEnvironment,
        "microwave": FrankaOpenMicrowaveEnvironment,
    }[args_cli.arena_task]
    arena_env = env_cls().get_env(env_args)
    name, cfg = ArenaEnvBuilder(arena_env, env_args).build_registered()
    env = gym.make(name, cfg=cfg).unwrapped
    return env, arena_env


def _ee_pose(env, obs) -> tuple[list[float], list[float]]:
    if isinstance(obs, dict) and isinstance(obs.get("policy"), dict):
        policy = obs["policy"]
        if "eef_pos" in policy:
            pos = policy["eef_pos"][0].detach().cpu().tolist()
            quat = policy.get("eef_quat")
            quat_list = quat[0].detach().cpu().tolist() if quat is not None else []
            return pos, quat_list
    try:
        pos_t = env.scene["ee_frame"].data.target_pos_w[0, 0] - env.scene.env_origins[0]
        quat_t = env.scene["ee_frame"].data.target_quat_w[0, 0]
        return pos_t.detach().cpu().tolist(), quat_t.detach().cpu().tolist()
    except Exception:
        return [], []


def _task_state(arena_env, env) -> dict[str, Any]:
    task = arena_env.task
    pressable = getattr(task, "pressable_object", None)
    openable = getattr(task, "openable_object", None)
    out: dict[str, Any] = {}
    if pressable is not None:
        try:
            out["pressed"] = bool(pressable.is_pressed(env).item())
        except Exception:
            pass
    if openable is not None:
        try:
            out["openness"] = float(openable.get_openness(env).item())
            out["open"] = bool(openable.is_open(env).item())
        except Exception:
            pass
    return out


def _draw_trail(prev_pos: list[float] | None, pos: list[float]) -> None:
    if prev_pos is None or not pos:
        return
    try:
        from isaacsim.util.debug_draw import _debug_draw

        draw = _debug_draw.acquire_debug_draw_interface()
        draw.draw_lines([tuple(prev_pos)], [tuple(pos)], [(1.0, 0.2, 0.1, 1.0)], [4.0])
        draw.draw_points([tuple(pos)], [(1.0, 0.9, 0.1, 1.0)], [8.0])
    except Exception:
        pass


def _print_controls() -> None:
    print(
        "\nEE keyboard controls:\n"
        "  W/S: +Y/-Y    A/D: -X/+X    E/Q: +Z/-Z\n"
        "  Arrow keys: rotate X/Y      Z/X: rotate Z\n"
        "  C: close gripper            V: open gripper\n"
        "  Enter: mark important waypoint\n"
        "  R: reset env                Esc: save and quit\n",
        flush=True,
    )


def main() -> None:
    os.makedirs(args_cli.output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jsonl_path = os.path.join(args_cli.output_dir, f"{args_cli.arena_task}_ee_teleop_{stamp}.jsonl")
    csv_path = os.path.join(args_cli.output_dir, f"{args_cli.arena_task}_ee_teleop_{stamp}.csv")

    env_args = _env_args()
    env, arena_env = _build_env(env_args)
    obs, _ = env.reset()
    controller = KeyboardEEController()
    _print_controls()
    print(f"[INFO] Recording JSONL: {jsonl_path}", flush=True)
    print(f"[INFO] Recording CSV:   {csv_path}", flush=True)

    prev_pos = None
    with open(jsonl_path, "w", encoding="utf-8") as jf, open(csv_path, "w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(
            cf,
            fieldnames=[
                "step",
                "time",
                "keys",
                "mark",
                "action",
                "ee_x",
                "ee_y",
                "ee_z",
                "task_state",
            ],
        )
        writer.writeheader()

        try:
            for step in range(args_cli.max_steps):
                if not simulation_app.is_running() or controller.quit:
                    break
                if controller.consume_reset():
                    obs, _ = env.reset()
                    prev_pos = None
                    print("[INFO] Reset env", flush=True)

                action = controller.action(device=torch.device(env.device), action_shape=env.action_space.shape)
                obs, _, terminated, truncated, _ = env.step(action)
                mark = controller.consume_mark()
                ee_pos, ee_quat = _ee_pose(env, obs)
                state = _task_state(arena_env, env)
                record = {
                    "step": step,
                    "time": time.time(),
                    "keys": sorted(controller.keys),
                    "mark": mark,
                    "action": action[0].detach().cpu().tolist(),
                    "ee_pos": ee_pos,
                    "ee_quat": ee_quat,
                    "task_state": state,
                    "terminated": bool(terminated.any()),
                    "truncated": bool(truncated.any()),
                }
                jf.write(json.dumps(record) + "\n")
                jf.flush()
                writer.writerow(
                    {
                        "step": step,
                        "time": record["time"],
                        "keys": "+".join(record["keys"]),
                        "mark": int(mark),
                        "action": json.dumps(record["action"]),
                        "ee_x": ee_pos[0] if len(ee_pos) >= 3 else "",
                        "ee_y": ee_pos[1] if len(ee_pos) >= 3 else "",
                        "ee_z": ee_pos[2] if len(ee_pos) >= 3 else "",
                        "task_state": json.dumps(state),
                    }
                )
                if step % args_cli.trail_interval == 0 and ee_pos:
                    _draw_trail(prev_pos, ee_pos)
                    prev_pos = ee_pos
                if mark:
                    print(f"[MARK] step={step} ee_pos={ee_pos} state={state}", flush=True)
                if terminated.any() or truncated.any():
                    print(f"[INFO] Episode ended step={step} state={state}", flush=True)
        finally:
            controller.close()
            env.close()
    print(f"[INFO] Saved teleop trajectory: {jsonl_path}", flush=True)
    print(f"[INFO] Saved teleop CSV:        {csv_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
