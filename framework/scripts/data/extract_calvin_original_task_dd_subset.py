#!/usr/bin/env python3
"""Resumably extract the exact original task_D_D task set from remote CALVIN ZIP."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import shutil
import struct
import threading
import time
from typing import Any
from zipfile import ZipFile
import zlib

import numpy as np
import requests

PREFIX='task_D_D/'
EPISODE_KEYS=('robot_obs','scene_obs','rel_actions')
LOCAL_HEADER=struct.Struct('<IHHHHHIIIHH')
LOCAL_FILE_SIGNATURE=0x04034B50


def member_end(info:Any)->int:
    return int(info.header_offset)+LOCAL_HEADER.size+len(info.filename.encode())+len(info.extra)+int(info.compress_size)+64


def member_bytes_from_range(payload:bytes,range_start:int,info:Any)->bytes:
    offset=int(info.header_offset)-range_start
    signature,_version,_flags,method,_mtime,_mdate,_crc,_csize,_usize,name_len,extra_len=LOCAL_HEADER.unpack_from(payload,offset)
    if signature!=LOCAL_FILE_SIGNATURE: raise RuntimeError(f'Invalid local header for {info.filename}')
    start=offset+LOCAL_HEADER.size+name_len+extra_len
    compressed=payload[start:start+int(info.compress_size)]
    if len(compressed)!=int(info.compress_size): raise RuntimeError(f'Truncated member {info.filename}')
    data=compressed if method==0 else zlib.decompress(compressed,-zlib.MAX_WBITS) if method==8 else None
    if data is None: raise RuntimeError(f'Unsupported ZIP method {method}')
    if len(data)!=int(info.file_size) or zlib.crc32(data)&0xffffffff!=int(info.CRC): raise RuntimeError(f'Size/CRC mismatch {info.filename}')
    return data


def physical_groups(infos:list[Any],max_gap:int=1<<20,max_range_bytes:int|None=None)->list[list[Any]]:
    groups:list[list[Any]]=[]
    for info in sorted(infos,key=lambda value:int(value.header_offset)):
        gap=bool(groups and int(info.header_offset)-member_end(groups[-1][-1])>max_gap)
        large=bool(groups and max_range_bytes is not None and member_end(info)-int(groups[-1][0].header_offset)+1>max_range_bytes)
        if not groups or gap or large: groups.append([info])
        else: groups[-1].append(info)
    return groups


def select_segments(annotation: dict[str,Any], expected_counts: dict[str,int], split_manifest:dict[str,Any]) -> list[dict[str,Any]]:
    lookup={}
    for index,(task,interval) in enumerate(zip(annotation['language']['task'],annotation['info']['indx'])):
        key=(str(task),int(interval[0]),int(interval[1]))
        if key in lookup: raise RuntimeError(f'Duplicate official annotation key {key}')
        lookup[key]=index
    selected={}
    for partition in ('source','target','holdout'):
        for raw in split_manifest[partition]:
            task=str(raw['task'])
            if task not in expected_counts: continue
            key=(task,int(raw['start']),int(raw['end']))
            if key not in lookup: raise RuntimeError(f'Original task_D_D interval absent from official annotation: {key}')
            selected[key]={'task':task,'start':key[1],'end':key[2],'annotation_index':lookup[key],'split':'source'}
    rows=list(selected.values())
    counts=Counter(str(row['task']) for row in rows)
    if counts != Counter({str(k):int(v) for k,v in expected_counts.items()}):
        raise RuntimeError(f'Official annotation counts differ from original run: {dict(counts)} != {expected_counts}')
    intervals=[(int(row['start']),int(row['end'])) for row in rows]
    if len(intervals)!=len(set(intervals)):
        raise RuntimeError('Duplicate annotation intervals in original task selection')
    return sorted(rows,key=lambda row:(int(row['start']),str(row['task'])))


def valid_member(path:Path,info:Any,verify_crc:bool=True)->bool:
    if not path.is_file() or path.stat().st_size!=int(info.file_size): return False
    if not verify_crc: return True
    crc=0
    with path.open('rb') as handle:
        while chunk:=handle.read(4<<20): crc=zlib.crc32(chunk,crc)
    return crc & 0xffffffff == int(info.CRC)


def atomic_write(path:Path,data:bytes)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+f'.tmp.{os.getpid()}.{threading.get_ident()}')
    tmp.write_bytes(data); os.replace(tmp,path)


def atomic_json(path:Path,payload:dict[str,Any])->None:
    atomic_write(path,(json.dumps(payload,indent=2,default=str)+'\n').encode())


def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--url',default='http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip')
    ap.add_argument('--zip-index',type=Path,required=True)
    ap.add_argument('--full-annotation',type=Path,required=True)
    ap.add_argument('--metadata-root',type=Path,required=True)
    ap.add_argument('--original-run-meta',type=Path,required=True)
    ap.add_argument('--source-split-manifest',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--reuse-data-root',type=Path,nargs='*',default=[])
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--retries',type=int,default=6)
    ap.add_argument('--timeout',type=float,default=120.0)
    ap.add_argument('--max-range-mib',type=int,default=32)
    ap.add_argument('--max-groups',type=int,default=0,help='Process at most N missing range groups, then exit cleanly for scheduled resume')
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    if args.workers<1 or args.retries<1: ap.error('workers and retries must be positive')

    started=time.time(); root=args.output_dir.resolve(); split=root/'training'
    meta=json.loads(args.original_run_meta.read_text())
    expected={str(k):int(v) for k,v in meta['actual_segments_by_task'].items()}
    annotation=np.load(args.full_annotation.resolve(),allow_pickle=True).item()
    split_manifest=json.loads(args.source_split_manifest.read_text())
    segments=select_segments(annotation,expected,split_manifest)
    selected_frames:set[int]=set()
    for row in segments:
        for frame in range(int(row['start']),int(row['end'])+1):
            selected_frames.add(frame)

    with ZipFile(args.zip_index.resolve()) as archive:
        infos={frame:archive.getinfo(f'{PREFIX}training/episode_{frame:07d}.npz') for frame in selected_frames}
        compressed_bytes=sum(int(info.compress_size) for info in infos.values())
        summary={'segments':len(segments),'segments_by_task':dict(sorted(expected.items())),'frames':len(infos),'compressed_bytes':compressed_bytes,'tasks':sorted(expected)}
        if len(segments)!=int(meta['actual_segments']): raise RuntimeError('Original total segment count mismatch')
        if args.dry_run:
            print(json.dumps(summary,indent=2)); return

        progress=root/'extract_progress.json'; split.mkdir(parents=True,exist_ok=True)
        reused_files=reused_bytes=0
        reuse_splits=[p.resolve()/'training' for p in args.reuse_data_root]
        for frame,info in infos.items():
            dest=split/f'episode_{frame:07d}.npz'
            # Completed downloads were CRC-checked before atomic rename; use a size-only
            # resume gate here and perform a full CRC sweep at the final completion gate.
            if valid_member(dest,info,verify_crc=False): continue
            for reuse in reuse_splits:
                src=reuse/f'episode_{frame:07d}.npz'
                if valid_member(src,info):
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    if dest.exists(): dest.unlink()
                    try: os.link(src,dest)
                    except OSError: shutil.copy2(src,dest)
                    reused_files+=1; reused_bytes+=int(info.file_size); break

        missing_infos=[info for frame,info in infos.items() if not valid_member(split/f'episode_{frame:07d}.npz',info,verify_crc=False)]
        all_missing_groups=physical_groups(missing_infos,max_range_bytes=args.max_range_mib<<20)
        missing_groups=all_missing_groups[:args.max_groups] if args.max_groups>0 else all_missing_groups

        downloaded_bytes=downloaded_files=completed_groups=0
        def fetch_group(group:list[Any])->tuple[int,int]:
            range_start=int(group[0].header_offset); range_end=member_end(group[-1]); session=requests.Session(); response=None
            for attempt in range(1,args.retries+1):
                try:
                    response=session.get(args.url,headers={'Range':f'bytes={range_start}-{range_end}'},timeout=args.timeout)
                    response.raise_for_status()
                    if response.status_code!=206: raise RuntimeError(f'Range ignored: {response.status_code}')
                    break
                except (requests.RequestException,RuntimeError):
                    if attempt==args.retries: raise
                    time.sleep(min(2**(attempt-1),30))
            assert response is not None
            for info in group:
                data=member_bytes_from_range(response.content,range_start,info)
                atomic_write(split/Path(info.filename).name,data)
            return len(response.content),len(group)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for nbytes,nfiles in executor.map(fetch_group,missing_groups):
                downloaded_bytes+=nbytes; downloaded_files+=nfiles; completed_groups+=1
                receipt={**summary,'complete':False,'missing_files_at_start':len(missing_infos),'missing_groups_at_start':len(all_missing_groups),'scheduled_groups_this_run':len(missing_groups),'completed_missing_groups_this_run':completed_groups,'downloaded_bytes_this_run':downloaded_bytes,'downloaded_files_this_run':downloaded_files,'reused_files_this_run':reused_files,'reused_bytes_this_run':reused_bytes,'elapsed_seconds':time.time()-started}
                atomic_json(progress,receipt)
                if completed_groups==1 or completed_groups==len(missing_groups) or completed_groups%25==0:
                    print(f'[extract] {completed_groups}/{len(missing_groups)} downloaded_gib={downloaded_bytes/2**30:.3f}',flush=True)

        if len(missing_groups)<len(all_missing_groups):
            receipt={**summary,'complete':False,'batch_complete':True,'remaining_groups_before_rescan':len(all_missing_groups)-len(missing_groups),'scheduled_groups_this_run':len(missing_groups),'completed_missing_groups_this_run':completed_groups,'downloaded_bytes_this_run':downloaded_bytes,'downloaded_files_this_run':downloaded_files,'elapsed_seconds':time.time()-started}
            atomic_json(progress,receipt)
            print(f'[extract] bounded batch complete; remaining_groups_before_rescan={receipt["remaining_groups_before_rescan"]}',flush=True)
            return

        missing=[]
        for frame,info in infos.items():
            path=split/f'episode_{frame:07d}.npz'
            if not valid_member(path,info,verify_crc=False):
                if path.exists(): path.unlink()
                missing.append(frame); continue
        sample_frames={int(row['start']) for row in segments}|{int(row['end']) for row in segments}
        for frame in sorted(sample_frames):
            path=split/f'episode_{frame:07d}.npz'
            with np.load(path,allow_pickle=True) as payload:
                absent=[key for key in EPISODE_KEYS if key not in payload]
                if absent: raise RuntimeError(f'{path} missing fields {absent}')
        if missing: raise RuntimeError(f'{len(missing)} final members failed size/CRC validation')

    metadata_split=args.metadata_root.resolve()/'training'
    (split/'.hydra').mkdir(parents=True,exist_ok=True)
    shutil.copy2(metadata_split/'.hydra/merged_config.yaml',split/'.hydra/merged_config.yaml')
    shutil.copy2(metadata_split/'scene_info.npy',split/'scene_info.npy')
    lang=split/'lang_annotations'; lang.mkdir(parents=True,exist_ok=True)
    pos=[int(row['annotation_index']) for row in segments]
    np.save(lang/'auto_lang_ann.npy',{'language':{k:np.asarray(v,dtype=object)[pos] if k!='emb' else np.asarray(v)[pos] for k,v in annotation['language'].items()},'info':{k:np.asarray(v)[pos] for k,v in annotation['info'].items()}},allow_pickle=True)
    manifest={**summary,'source_url':args.url,'original_run_meta':args.original_run_meta.resolve(),'source_split_manifest':args.source_split_manifest.resolve(),'segments_detail':segments,'downloaded_bytes_this_run':downloaded_bytes,'downloaded_files_this_run':downloaded_files,'reused_files_this_run':reused_files,'reused_bytes_this_run':reused_bytes,'missing_frames':[],'elapsed_seconds':time.time()-started,'complete':True}
    atomic_json(root/'original_task_dd_manifest.json',manifest); atomic_json(progress,manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k!='segments_detail'},indent=2,default=str))

if __name__=='__main__': main()
