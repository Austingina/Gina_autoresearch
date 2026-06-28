"""Introspect ManiSkill cabinet joint to find world-frame axis + hinge origin.

We need, per target joint:
  - prismatic: world slide axis (unit) + sign toward 'open'
  - revolute : world hinge axis (unit) + a point on the hinge (origin) + sign

Prints candidate attributes so we wire the ACTUATE κ direction against the
real API instead of guessing.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import mani_skill  # noqa: F401


def show(env_id: str) -> None:
    print(f"\n================ {env_id} ================")
    env = gym.make(env_id, num_envs=1, obs_mode="state", sim_backend="physx_cuda")
    base = env.unwrapped
    env.reset(seed=0)
    j = base.handle_link.joint
    link = base.handle_link
    print("joint repr:", type(j).__name__)
    attrs = [a for a in dir(j) if not a.startswith("__")]
    print("joint attrs:", attrs)
    for name in [
        "type", "axis", "pose_in_parent", "pose_in_child",
        "global_pose", "qpos", "limits", "name",
    ]:
        if hasattr(j, name):
            try:
                v = getattr(j, name)
                if torch.is_tensor(v):
                    print(f"  j.{name}: tensor {tuple(v.shape)} -> {v.flatten()[:8].cpu().numpy()}")
                elif hasattr(v, "raw_pose"):
                    print(f"  j.{name}.raw_pose: {v.raw_pose.flatten().cpu().numpy()}")
                else:
                    print(f"  j.{name}: {v}")
            except Exception as e:
                print(f"  j.{name}: <err {e}>")

    print("-- link pose --")
    try:
        print("  link.pose.raw_pose:", link.pose.raw_pose.flatten().cpu().numpy())
    except Exception as e:
        print("  link.pose err", e)

    # finite-difference ground truth: move target joint toward open, see handle delta
    print("-- finite-diff handle motion (open by small dq) --")
    try:
        hp0 = base.handle_link_positions().clone()
        q = base.cabinet.get_qpos().clone()
        # nudge ALL dofs a touch toward their target open via qpos set on the cabinet's target joint
        qlim = base.cabinet.get_qlimits()
        # find which dof index the handle joint corresponds to is non-trivial; instead set whole qpos
        q_new = q.clone()
        # open everything slightly toward upper limit
        q_new = q + 0.05 * (qlim[..., 1] - qlim[..., 0])
        base.cabinet.set_qpos(q_new)
        if base.gpu_sim_enabled:
            base.scene._gpu_apply_all()
            base.scene.px.gpu_update_articulation_kinematics()
            base.scene._gpu_fetch_all()
        hp1 = base.handle_link_positions().clone()
        d = (hp1 - hp0)[0].cpu().numpy()
        n = float(np.linalg.norm(d))
        print(f"  handle delta = {d}  |  unit = {d / n if n > 1e-6 else d}  (norm={n:.4f})")
    except Exception as e:
        import traceback

        traceback.print_exc()
        print("  finite-diff err", e)
    env.close()


if __name__ == "__main__":
    for e in ["OpenCabinetDrawer-v1", "OpenCabinetDoor-v1"]:
        show(e)
