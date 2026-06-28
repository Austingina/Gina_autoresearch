"""ManiSkill3 smoke test for the Interaction Geometry task set.

Checks, per task:
  - env builds + GPU sim steps
  - obs_mode="rgb+depth+segmentation" produces seg + depth tensors
  - prints sensor names / shapes / dtypes and key interaction handles

Tasks (embrace_kappa route):
  PickCube-v1            (Panda,  free          κ=vertical_free)
  OpenCabinetDrawer-v1   (Fetch,  prismatic     κ=slide)
  OpenCabinetDoor-v1     (Fetch,  revolute      κ=revolute)
"""

from __future__ import annotations

import argparse
import traceback

import gymnasium as gym
import numpy as np
import torch

import mani_skill  # noqa: F401  (registers envs)


def _describe(x, name: str, indent: str = "      ") -> None:
    if isinstance(x, dict):
        print(f"{indent}{name}: dict[{', '.join(x.keys())}]")
        for k, v in x.items():
            _describe(v, k, indent + "  ")
    elif torch.is_tensor(x):
        uniq = ""
        if x.numel() and x.dtype in (torch.int16, torch.int32, torch.int64, torch.uint8):
            try:
                uniq = f" uniq={torch.unique(x).cpu().numpy()[:8].tolist()}"
            except Exception:
                uniq = ""
        print(f"{indent}{name}: tensor {tuple(x.shape)} {x.dtype} dev={x.device}{uniq}")
    elif isinstance(x, np.ndarray):
        print(f"{indent}{name}: ndarray {x.shape} {x.dtype}")
    else:
        print(f"{indent}{name}: {type(x).__name__}")


def smoke_one(env_id: str, num_envs: int, obs_mode: str) -> bool:
    print(f"\n=== {env_id} (num_envs={num_envs}, obs_mode={obs_mode}) ===")
    try:
        env = gym.make(env_id, num_envs=num_envs, obs_mode=obs_mode, sim_backend="physx_cuda")
    except Exception:
        traceback.print_exc()
        print(f"[FAIL build] {env_id}")
        return False
    try:
        base = env.unwrapped
        print(f"  robot_uids   = {getattr(base, 'robot_uids', '?')}")
        print(f"  action_space = {env.action_space}")
        obs, info = env.reset(seed=0)
        print("  -- obs structure --")
        _describe(obs, "obs")
        act = env.action_space.sample()
        for _ in range(3):
            obs, rew, term, trunc, info = env.step(act)
        print("  -- info keys --", list(info.keys()))
        if "success" in info:
            print("  success:", info["success"].cpu().numpy().tolist() if torch.is_tensor(info["success"]) else info["success"])
        # interaction handles for cabinet tasks
        if hasattr(base, "handle_link_positions"):
            hp = base.handle_link_positions()
            print("  handle_link_pos:", tuple(hp.shape), "sample:", hp[0].cpu().numpy())
            j = base.handle_link.joint
            print("  joint.type:", j.type, "target_qpos:", base.target_qpos.flatten()[:num_envs].cpu().numpy())
        print(f"[OK] {env_id}")
        return True
    except Exception:
        traceback.print_exc()
        print(f"[FAIL run] {env_id}")
        return False
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_envs", type=int, default=2)
    ap.add_argument("--obs_mode", type=str, default="rgb+depth+segmentation")
    ap.add_argument(
        "--envs",
        type=str,
        nargs="+",
        default=["PickCube-v1", "OpenCabinetDrawer-v1", "OpenCabinetDoor-v1"],
    )
    args = ap.parse_args()

    print("torch", torch.__version__, "cuda_ok", torch.cuda.is_available())
    results = {e: smoke_one(e, args.num_envs, args.obs_mode) for e in args.envs}
    print("\n=== SUMMARY ===")
    for e, ok in results.items():
        print(f"  {'OK ' if ok else 'FAIL'}  {e}")


if __name__ == "__main__":
    main()
