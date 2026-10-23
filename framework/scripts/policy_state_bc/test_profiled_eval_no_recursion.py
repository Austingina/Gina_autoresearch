#!/usr/bin/env python3
"""Regression: profiling and rollout wrappers must not recurse."""
from __future__ import annotations
import importlib.util, json, tempfile
from pathlib import Path

REPO=Path(__file__).resolve().parents[3]
POLICY=REPO/'framework/scripts/policy_state_bc/train_eval_state_chunk_flow_policy.py'
CALVIN=Path('/home/ai-sz-a26394-u1/AST_runtime/AST_CALVIN_EXPAND_V1/calvin_g/code/calvin')
RUN=REPO/'experiments/phase1_calvin/2026-10-20_same_g_coverage_comparison/formal/old_legacy'
VIEW=REPO/'experiments/phase1_calvin/2026-10-20_same_g_full_formal/views/old_legacy'


def main():
    spec=importlib.util.spec_from_file_location('chunk_flow',POLICY); module=importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(module)
    meta=json.loads((RUN/'run_meta.json').read_text()); tasks={str(k):int(v) for k,v in meta['tasks'].items()}
    text_to_task,_=module.base.load_eval_annotation_maps(CALVIN)
    with tempfile.TemporaryDirectory() as tmp:
        summary=module.evaluate_policy_ckpt(CALVIN,VIEW,Path(tmp),RUN/'chunk_bc_minimal_g_concat/policy.pt',tasks,text_to_task,1,2,'cuda:0','balanced_first_task',{'close_drawer'},200,1,0,None,profile_eval_timing=True,state_only_observation=True)
    assert len(summary['raw_results'])==1
    print('profiled CALVIN rollout recursion regression: OK')

if __name__=='__main__': main()
