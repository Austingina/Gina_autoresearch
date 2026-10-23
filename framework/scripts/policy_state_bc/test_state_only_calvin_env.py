#!/usr/bin/env python3
"""Integration regression: state-only CALVIN eval must not instantiate cameras/tacto."""
from __future__ import annotations
import importlib.util
from pathlib import Path

REPO=Path(__file__).resolve().parents[3]
SCRIPT=REPO/'framework/scripts/policy_state_bc/train_eval_state_chunk_flow_policy.py'


def main():
    spec=importlib.util.spec_from_file_location('chunk_flow',SCRIPT)
    module=importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(module)
    root=REPO/'experiments/phase1_calvin/2026-10-20_same_g_full_formal/views/old_legacy'
    env=module.make_state_only_calvin_env(root)
    obs=env.reset()
    assert obs['robot_obs'].shape==(15,)
    assert obs['scene_obs'].shape==(24,)
    assert env.cameras==[]
    print('state-only CALVIN environment: OK')

if __name__=='__main__': main()
