"""Validate analytic ACTUATE G* (κ) against finite-difference ground truth.

For drawer/door: compute analytic G*_actuate from joint pose, then nudge the
target joint toward 'open' and measure the true handle displacement. Report
cosine similarity (should be ~1.0). Also prints REACH sanity for PickCube.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import mani_skill  # noqa: F401

from goals.maniskill_bridge import env_kappa, extract_interaction_state, oracle_actuate_field


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def validate_cabinet(env_id: str, seeds=(0, 1, 2, 3), num_envs: int = 4) -> None:
    print(f"\n==== {env_id}  (κ={env_kappa(env_id).value}) ====")
    env = gym.make(env_id, num_envs=num_envs, obs_mode="state", sim_backend="physx_cuda")
    base = env.unwrapped
    kappa = env_kappa(env_id)
    cos_all = []
    for s in seeds:
        env.reset(seed=s)
        state = extract_interaction_state(env, env_id)
        g_analytic = oracle_actuate_field(state, kappa, open_sign=1.0)  # (B,3)

        hp0 = base.handle_link_positions().clone()
        q = base.cabinet.get_qpos().clone()
        qlim = base.cabinet.get_qlimits()
        q_new = q + 0.04 * (qlim[..., 1] - qlim[..., 0])
        base.cabinet.set_qpos(q_new)
        if base.gpu_sim_enabled:
            base.scene._gpu_apply_all()
            base.scene.px.gpu_update_articulation_kinematics()
            base.scene._gpu_fetch_all()
        hp1 = base.handle_link_positions().clone()
        d = _np(hp1 - hp0)  # (B,3) finite-diff motion toward open

        for i in range(num_envs):
            n = np.linalg.norm(d[i])
            if n < 1e-5:
                continue
            fd = d[i] / n
            cos = float(np.dot(g_analytic[i], fd))
            cos_all.append(cos)
    cos_all = np.array(cos_all)
    print(f"  N={len(cos_all)}  cos(analytic, finite-diff): mean={cos_all.mean():.4f} "
          f"min={cos_all.min():.4f} max={cos_all.max():.4f}")
    print(f"  verdict: {'PASS' if cos_all.mean() > 0.97 else 'CHECK'}")
    env.close()


def sanity_pickcube() -> None:
    print("\n==== PickCube-v1  (κ=vertical_free) ====")
    env = gym.make("PickCube-v1", num_envs=2, obs_mode="state", sim_backend="physx_cuda")
    env.reset(seed=0)
    st = extract_interaction_state(env, "PickCube-v1")
    from goals.maniskill_bridge import oracle_actuate_field, oracle_reach_field

    print("  reach G* (ee->cube):", oracle_reach_field(st)[0])
    print("  actuate G* (vertical_free):", oracle_actuate_field(st, env_kappa("PickCube-v1"))[0],
          "(expect ~[0,0,1])")
    env.close()


if __name__ == "__main__":
    sanity_pickcube()
    validate_cabinet("OpenCabinetDrawer-v1")
    validate_cabinet("OpenCabinetDoor-v1")
