#!/usr/bin/env python3
"""Contract test for exact original task_D_D sparse selection."""
from __future__ import annotations
from collections import Counter
import importlib.util, json
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[3]
TARGET=REPO/'framework/scripts/data/extract_calvin_original_task_dd_subset.py'
ANN=REPO/'artifacts/calvin_zip_index/task_D_D_metadata/training/lang_annotations/auto_lang_ann.npy'
META=Path('/home/ai-sz-a26394-u1/AutoResearch/experiments/phase1_calvin/2026-07-13_minimal_g_projected_firstw16_e50_n100_h160_seed0/run_meta.json')
SPLIT=Path('/home/ai-sz-a26394-u1/AutoResearch/experiments/phase1_calvin/task_D_D_reusable_g_v1_object_transfer_red_blue_to_pink/artifacts/object_transfer_red_blue_to_pink.json')


def main():
    spec=importlib.util.spec_from_file_location('extract_original',TARGET)
    module=importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(module)
    meta=json.loads(META.read_text()); expected={str(k):int(v) for k,v in meta['actual_segments_by_task'].items()}
    annotation=np.load(ANN,allow_pickle=True).item()
    rows=module.select_segments(annotation,expected,json.loads(SPLIT.read_text()))
    assert len(rows)==meta['actual_segments']==2107
    assert Counter(r['task'] for r in rows)==Counter(expected)
    intervals=[(r['start'],r['end']) for r in rows]
    assert len(intervals)==len(set(intervals))
    assert all(a<=b for a,b in intervals)
    print('original task_D_D exact selection: OK')

if __name__=='__main__': main()
