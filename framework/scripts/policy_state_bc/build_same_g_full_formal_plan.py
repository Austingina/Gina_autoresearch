#!/usr/bin/env python3
"""Build audited dataset views and a fixed plan for the full same-G experiment."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def segment_sets(manifest_path: Path) -> tuple[set[tuple[str,int,int]],set[tuple[str,int,int]]]:
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    source=set(); held=set()
    for row in manifest.get('segments_detail',[]):
        key=(str(row['task']),int(row['start']),int(row['end']))
        split=str(row['split'])
        if split=='source': source.add(key)
        elif split=='heldout': held.add(key)
        else: raise RuntimeError(f'Unknown split {split!r} in {manifest_path}')
    if not source or not held or source & held:
        raise RuntimeError(f'Invalid source/heldout separation in {manifest_path}')
    return source,held


def ensure_view(view: Path, training_source: Path, validation_config: Path) -> None:
    view.mkdir(parents=True,exist_ok=True)
    link=view/'training'
    target=training_source.resolve()
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve()!=target:
            raise RuntimeError(f'Unexpected existing training view: {link}')
    else:
        link.symlink_to(target, target_is_directory=True)
    destination=view/'validation/.hydra/merged_config.yaml'
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if destination.read_bytes()!=validation_config.read_bytes():
            raise RuntimeError(f'Validation config mismatch: {destination}')
    else:
        tmp=destination.with_suffix('.yaml.tmp'); shutil.copyfile(validation_config,tmp); os.replace(tmp,destination)


def build_plan(repo: Path, output_root: Path, validation_config: Path) -> dict[str,Any]:
    repo=repo.resolve(); output_root=output_root.resolve(); validation_config=validation_config.resolve()
    old_root=repo/'data/calvin/task_ABC_D_g_intent_factorial_s12_h3'
    new_root=repo/'data/calvin/task_ABC_D_g_intent_multifold_s12'
    specs=[('old_legacy','old_legacy',old_root,old_root/'intent_factorial_manifest.json')]
    folds=sorted((repo/'artifacts/data_distribution/intent_multifold/folds').glob('*.json'))
    if len(folds)!=12: raise RuntimeError(f'Expected 12 fold manifests, found {len(folds)}')
    specs += [(f'new_{p.stem}','new_multifold',new_root,p) for p in folds]
    conditions=[]
    for name,kind,data_root,manifest in specs:
        source,held=segment_sets(manifest)
        view=output_root/'views'/name
        ensure_view(view,data_root/'training',validation_config)
        conditions.append({
            'name':name,'kind':kind,'view':view.as_posix(),
            'training_source':(data_root/'training').resolve().as_posix(),
            'manifest':manifest.resolve().as_posix(),
            'manifest_sha256':sha256(manifest),
            'source_segments':len(source),'heldout_segments':len(held),
            'source_tasks':sorted({x[0] for x in source}),
            'heldout_tasks':sorted({x[0] for x in held}),
        })
    return {
        'format':'same_g_full_formal_plan_v1',
        'validation_config':validation_config.as_posix(),
        'validation_config_sha256':sha256(validation_config),
        'shared_training':{
            'mode':'chunk_bc_minimal_g_concat','epochs':50,'max_windows':0,'stride':2,
            'chunk_size':16,'batch_size':256,'hidden':256,'layers':3,'seed':20260814,
            'executor_state_scope':'proprio','executor_language_mode':'none',
            'minimal_g_scope':'translational',
        },
        'conditions':conditions,
    }


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',type=Path,required=True); ap.add_argument('--output-root',type=Path,required=True); ap.add_argument('--validation-config',type=Path,required=True); ap.add_argument('--plan-output',type=Path,required=True); args=ap.parse_args()
    plan=build_plan(args.repo,args.output_root,args.validation_config)
    args.plan_output.parent.mkdir(parents=True,exist_ok=True)
    if args.plan_output.exists(): raise RuntimeError(f'Refusing to overwrite plan: {args.plan_output}')
    tmp=args.plan_output.with_suffix('.json.tmp'); tmp.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8'); os.replace(tmp,args.plan_output)
    print(json.dumps({'conditions':len(plan['conditions']),'plan':args.plan_output.as_posix(),'validation_sha256':plan['validation_config_sha256']},indent=2))

if __name__=='__main__': main()
