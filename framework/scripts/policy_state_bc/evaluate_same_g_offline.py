#!/usr/bin/env python3
"""Offline held-out first-action audit for the same-G coverage experiment."""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
from typing import Any
import numpy as np
import torch


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("chunk_flow", path)
    module = importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(module)
    return module


def summarize(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "first_action_mse": float(np.mean((pred - target) ** 2)),
        "first_xyz_mse": float(np.mean((pred[:, :3] - target[:, :3]) ** 2)),
        "first_gripper_acc": float(np.mean(np.sign(pred[:, 6]) == np.sign(target[:, 6]))),
    }


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--run-dir",type=Path,required=True); ap.add_argument("--data-root",type=Path,required=True); ap.add_argument("--selection-manifest",type=Path,required=True); ap.add_argument("--script",type=Path,required=True); ap.add_argument("--articulation-geometry",type=Path,required=True); ap.add_argument("--primitive-geometry",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--device",default="cuda:0"); ap.add_argument("--max-windows",type=int,default=0); ap.add_argument("--stride",type=int,default=4); args=ap.parse_args()
    m=load_module(args.script)
    run_meta=json.loads((args.run_dir/"run_meta.json").read_text()); task_to_id={k:int(v) for k,v in run_meta["tasks"].items()}
    all_segments=m.base.load_lang_segments(args.data_root/"training")
    chosen,receipt=m.select_segments_from_manifest(all_segments,args.selection_manifest,"heldout")
    keys={m.segment_key(s) for s in chosen}
    articulation=json.loads(args.articulation_geometry.read_text()); primitive,_=m.load_fixed_primitive_geometry(args.primitive_geometry,{"push_into_drawer","place_in_slider"})
    arrays,_=m.collect_chunk_records(args.data_root,task_to_id,16,args.max_windows,0,args.stride,20260814,None,articulation,"translational",primitive,packed_state_arrays_name="state_only_arrays.npz",selected_segment_keys=keys)
    target=np.asarray(arrays["action_chunks"][:,0,:],dtype=np.float32); tasks=np.asarray(arrays["task_name"],dtype=object)
    text_to_task={task:task for task in task_to_id}; ckpt=args.run_dir/"chunk_bc_minimal_g_concat/policy.pt"
    predictions: dict[str,np.ndarray]={}
    for control in ("normal","zero","sign_flip"):
        policy=m.ChunkFlowPolicy(ckpt,task_to_id,text_to_task,args.device,query_g_control=control)
        rows=[]
        for robot,scene,task in zip(arrays["robot_obs"],arrays["scene_obs"],tasks):
            policy.reset(); rows.append(policy.step({"robot_obs":robot,"scene_obs":scene},str(task)))
        predictions[control]=np.stack(rows)
    report: dict[str,Any]={"format":"same_g_offline_heldout_v1","selection":receipt,"windows":int(len(target)),"normal":summarize(predictions["normal"],target),"controls":{}}
    for control in ("zero","sign_flip"):
        report["controls"][control]={**summarize(predictions[control],target),"prediction_delta_l2":float(np.mean(np.linalg.norm(predictions[control]-predictions["normal"],axis=1)))}
    report["per_task"]={}
    for task in sorted(set(map(str,tasks))):
        mask=tasks==task; report["per_task"][task]={"windows":int(mask.sum()),**summarize(predictions["normal"][mask],target[mask])}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))

if __name__=="__main__": main()
