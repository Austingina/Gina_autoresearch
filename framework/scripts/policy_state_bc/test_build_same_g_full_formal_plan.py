#!/usr/bin/env python3
"""Contract test for the full same-G formal experiment plan."""
from __future__ import annotations
import importlib.util, json
from pathlib import Path
import tempfile

REPO=Path(__file__).resolve().parents[3]
SCRIPT=REPO/'framework/scripts/policy_state_bc/build_same_g_full_formal_plan.py'


def main():
    spec=importlib.util.spec_from_file_location('formal_plan',SCRIPT)
    module=importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(module)
    validation=REPO/'experiments/phase1_calvin/2026-10-20_same_g_full_formal/common/official_validation/.hydra/merged_config.yaml'
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        plan=module.build_plan(REPO,root,validation)
        assert len(plan['conditions'])==13
        assert plan['shared_training']['epochs']==50
        assert plan['shared_training']['max_windows']==0
        assert plan['shared_training']['stride']==2
        assert plan['shared_training']['seed']==20260814
        assert plan['conditions'][0]['name']=='old_legacy'
        assert sum(c['kind']=='new_multifold' for c in plan['conditions'])==12
        for condition in plan['conditions']:
            view=Path(condition['view'])
            assert (view/'training').is_symlink()
            assert (view/'training').resolve()==Path(condition['training_source']).resolve()
            assert (view/'validation/.hydra/merged_config.yaml').read_bytes()==validation.read_bytes()
            manifest=json.loads(Path(condition['manifest']).read_text())
            source={(x['task'],int(x['start']),int(x['end'])) for x in manifest['segments_detail'] if x['split']=='source'}
            held={(x['task'],int(x['start']),int(x['end'])) for x in manifest['segments_detail'] if x['split']=='heldout'}
            assert source and held and source.isdisjoint(held)
    print('same-G full formal plan: OK')

if __name__=='__main__': main()
