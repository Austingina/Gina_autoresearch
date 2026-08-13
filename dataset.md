# Dataset ledger

This file is the canonical record of every data source needed to reproduce the
results cited by `PROGRESS.md`. Large files are deliberately not committed to
Git. Paths below are examples; scripts should use the environment variables in
`bootstrap.md` instead of depending on one machine's mount point.

Last audited against the BNU V100 migration workspace on 2026-08-14.

## Recovery scope

The current paper path has two experiment phases:

| Phase | Data | Status |
| --- | --- | --- |
| Phase 0, Isaac Lab lift | Project-collected HDF5 demonstrations | Legacy evidence; regenerate with the vendored teacher |
| Phase 1, CALVIN | Official debug, `task_D_D`, and `task_ABC_D` datasets | Active evidence and ABC-to-D visual scene gate |

RoboTwin is not a dependency of the current experiment path. Old RoboTwin
notes may remain as historical context, but no RoboTwin dataset is required by
the bootstrap procedure.

## Canonical storage layout

```text
$AUTORESEARCH_ROOT/                         # Git checkout
$CALVIN_G_ROOT/
  code/calvin/                              # official CALVIN checkout
  envs/calvin_h200/                         # Python environment, not in Git
  data/
    calvin_debug_dataset/                   # early smoke tests
    task_D_D/                               # D demonstrations and D eval config
    task_ABC_D.zip                          # official A/B/C -> D archive
    task_ABC_D_scene_info_fix.zip           # official metadata repair archive
    task_ABC_D_compact_blocks/              # derived state/action subset
    task_ABC_D_compact_blocks_dinov2_small/ # derived frozen visual features
    task_ABC_D_volume_g_subset/              # 1.7 GB range-extracted method gate
      training/state_only_arrays.npz         # 1.25 MB matched volume-G cache
    lerobot_blocks_behavior_g_semantic/     # diagnostic six-task behavior G
      training/state_only_arrays.npz        # packed state/action/G cache
    lerobot_blocks_behavior_g_semantic_dinov2_small/
```

The audited H200 paths were:

```text
AUTORESEARCH_ROOT=/mnt/nvme33n1/wuk_code/AutoResearch
CALVIN_G_ROOT=/mnt/nvme33n1/wuk_code/calvin_g
```

## Source 1: official CALVIN data

Project and license:

- Repository: <https://github.com/mees/calvin>
- Dataset page: <https://calvin.cs.uni-freiburg.de/>
- CALVIN commit used here: `fa03f01f19c65920e18cf37398a9ce859274af76`
- `calvin_env` submodule commit: `1431a46bd36bde5903fb6345e68b5ccc30def666`
- License and citation are in the official repository.

Direct official downloads:

```text
http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip
http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip
http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip
http://calvin.cs.uni-freiburg.de/scene_info_fix/task_D_D_scene_info.zip
http://calvin.cs.uni-freiburg.de/scene_info_fix/task_ABC_D_scene_info.zip
```

The official wrapper exposes the same downloads:

```bash
cd "$CALVIN_CODE/dataset"
bash download_data.sh debug
bash download_data.sh D
bash download_data.sh ABC
```

For the 500+ GB archive, use a resumable transfer instead of the wrapper's
plain `wget`:

```bash
mkdir -p "$CALVIN_DATA"
cd "$CALVIN_DATA"
wget -c http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip
```

### Files actually used

| Item | Audited size | Role |
| --- | ---: | --- |
| `calvin_debug_dataset/` | 1.3 GB | Pipeline and dual-camera smoke tests |
| `task_D_D/` | 168 GB extracted | Early D-only experiments and current D closed-loop evaluation |
| `task_ABC_D.zip` | 518 GB | A/B/C source demonstrations, source RGB, language annotations |
| `task_ABC_D_scene_info_fix.zip` | 525 B | Official corrected `training/scene_info.npy` |

Audited SHA-256 values:

```text
c2036c67eb4c06966af1d1e1665bdb572c69e1404f5e77ffd46b384ff2b79f74  task_ABC_D.zip
3c13651947dc4927b1df3b700143e1d47605db0958f296a0a691b911876d2a64  task_ABC_D_scene_info_fix.zip
471dbc647ea4822d7b0ebb1b289f1cc856eb6a6a0bc829e31d4a5876820310a6  task_D_D_scene_info.zip
```

The original `task_D_D.zip` was deleted after extraction and its project-local
hash was not retained. Recover it only from the official URL, apply the
official scene-info repair, and validate the extracted structure as shown
below. Do not substitute a converted Hugging Face/LeRobot mirror for formal
numbers because conversion can alter actions, annotations, or indexing.

### Short-window recovery mirror (diagnostic only)

When the official 518 GB server was too slow for the four-hour BNU allocation,
the method-validation path used the ModelScope mirror
`zhihelu/calvin-task-ABC-D`. This is a 57.84 GB LeRobot/RLDS conversion with
18,957 episodes and approximately 1.136 million frames:

```text
https://modelscope.cn/datasets/zhihelu/calvin-task-ABC-D
```

The archive is split into two Git LFS objects. Verify both before extraction:

```text
91b1a2083024b1d6445db3d2bbc974e3a6d90beeef6855aa3b793ada77e6ea4f  task-ABC-D.tar.gz.part00  (32212254720 bytes)
b7f29eb2eac3f8f10370a309640b9540b5e3f9deb1ad58bede7b9fe0fc7abf24  task-ABC-D.tar.gz.part01  (25623781844 bytes)
```

This conversion retains top RGB (200 x 200), wrist RGB (84 x 84), the exact
15-D `robot_obs` as `observation.state`, and 7-D relative actions. It does not
retain `scene_obs`. Consequently it can support visual/proprioceptive policy
training and video-derived G experiments, but it cannot reproduce an oracle
scene-state G or replace official ABC-to-D benchmark numbers.

For an early migration probe, episodes were filtered to the six block-push
tasks and converted with:

```bash
"$CALVIN_PYTHON" \
  framework/scripts/data/convert_calvin_lerobot_to_compact_npz.py \
  --lerobot-root "$CALVIN_DATA/modelscope_calvin_abc_extracted" \
  --meta-root "$CALVIN_DATA/hf_calvin_abc_lerobot/meta" \
  --camera-config-root "$CALVIN_DATA/calvin_debug_dataset" \
  --output-dir "$CALVIN_DATA/lerobot_blocks_behavior_g_semantic" \
  --include-tasks \
    push_red_block_left push_red_block_right \
    push_blue_block_left push_blue_block_right \
    push_pink_block_left push_pink_block_right \
  --flow-mode semantic_axis
```

The diagnostic G is intentionally coarse. Language determines the signed
world-x object-flow vector. Target-colored pixels and projected EEF motion
estimate contact onset; before contact, attraction points from EEF to the
episode-level contact anchor, and after contact attraction is zero. The script
writes `minimal_g=[attraction(3), flow(3)]` per frame. This is a behavior-field
probe derived from video/proprioception, not an oracle contact label.

The complete audited conversion has 3,357 episodes and 199,292 frames. Before
multi-seed training, pack the small numeric arrays into one verified cache:

```bash
"$CALVIN_PYTHON" \
  framework/scripts/data/pack_calvin_state_arrays.py \
  --data-root "$CALVIN_DATA/lerobot_blocks_behavior_g_semantic" \
  --workers 8
```

Expected output is
`training/state_only_arrays.npz` with `frame_index[199292]`,
`robot_obs[199292,15]`, `scene_obs[199292,24]`,
`rel_actions[199292,7]`, and `minimal_g[199292,6]`. The audited file is
43,048,352 bytes (about 42 MiB) and took 128 seconds to build on 20 CPU cores.
`packed_state_manifest.json` records shapes and completion. Training detects
this cache automatically; source per-frame files remain the provenance source
and are still required by the visual-feature alignment checks.

Apply metadata repairs to extracted datasets when needed:

```bash
cd "$CALVIN_DATA/task_D_D"
wget -c http://calvin.cs.uni-freiburg.de/scene_info_fix/task_D_D_scene_info.zip
unzip -o task_D_D_scene_info.zip

# Only needed when a full task_ABC_D directory is extracted. The compact
# builder below reads demonstrations and language annotations from the zip.
cd "$CALVIN_DATA/task_ABC_D"
wget -c http://calvin.cs.uni-freiburg.de/scene_info_fix/task_ABC_D_scene_info.zip
unzip -o task_ABC_D_scene_info.zip
```

Expected CALVIN split structure:

```text
task_D_D/{training,validation}/
  episode_*.npz
  lang_annotations/auto_lang_ann.npy
  .hydra/merged_config.yaml
task_D_D/{training,validation}/scene_info.npy
```

The previously audited D extraction contained 512,077 training frame files
and 99,022 validation frame files. These are time-step files, not 611,099
independent task demonstrations.

### Actions, observations, and language

The active executor reads CALVIN's `robot_obs`, `scene_obs`, and 7-D
`rel_actions`. Visual experiments read the official `rgb_static` and
`rgb_gripper` frames. The 384-D `paraphrase-MiniLM-L3-v2` language embeddings
are already included in CALVIN's annotation files; they are not generated by a
separate AutoResearch download.

Formal evaluation must preserve CALVIN's relative-action wrapper semantics.
Passing raw relative actions through an absolute-action environment interface
produces incorrect trajectories.

## Derived CALVIN data

Derived directories are disposable caches. They must be regenerated from the
official archive rather than copied into Git.

### Compact A/B/C block-push subset

The active ABC-to-D gate does not extract the full 518 GB archive. It selects
four source task combinations and stores only state, scene state, relative
action, and aligned language annotations:

```bash
"$CALVIN_PYTHON" \
  "$AUTORESEARCH_ROOT/framework/scripts/data/build_calvin_compact_state_dataset_from_zip.py" \
  --zip "$CALVIN_DATA/task_ABC_D.zip" \
  --out-dir "$CALVIN_DATA/task_ABC_D_compact_blocks" \
  --include-tasks \
    push_red_block_left \
    push_blue_block_right \
    push_pink_block_left \
    push_pink_block_right
```

The builder is resumable and writes both
`compact_extract_progress.json` and `compact_extract_manifest.json`.

Expected manifest invariants:

| Field | Expected value |
| --- | --- |
| segments | 2,101 |
| frames / episode files | 51,594 |
| red-left segments | 525 |
| blue-right segments | 526 |
| pink-left segments | 525 |
| pink-right segments | 525 |
| missing IDs | empty |
| compact disk use | approximately 207 MB |

Training sees every color and direction value but not the held-out
color-direction combinations `push_red_block_right` and
`push_blue_block_left`. Those combinations are evaluated in environment D.

### Range-extracted interaction-volume method gate

The current representation gate does not require the full 518 GB ABC archive.
The official server supports HTTP range requests, so the extractor reads ZIP
metadata and downloads only a deterministic, balanced 1.7 GB subset:

```bash
"$CALVIN_PYTHON" -m pip install remotezip requests

"$CALVIN_PYTHON" \
  "$AUTORESEARCH_ROOT/framework/scripts/data/extract_calvin_remote_calibration_subset.py" \
  --output-dir "$CALVIN_DATA/task_ABC_D_volume_g_subset"
```

Expected invariants are 114 non-overlapping segments and 6,810 frames: each of
the four source tasks has 24 segments, each held-out audit task has 9, and each
A/B/C scene has 38. The rebuilt annotation preserves official language text,
task labels, and 384-D embeddings. The selection seed is `20260814`; full
segment indices and provenance are in `calibration_subset_manifest.json`.

Fit geometry using source tasks only, then write matched per-frame G labels:

```bash
"$CALVIN_PYTHON" \
  "$AUTORESEARCH_ROOT/framework/scripts/data/calibrate_calvin_block_surface_g.py" \
  --data-root "$CALVIN_DATA/task_ABC_D_volume_g_subset" \
  --calvin-code "$CALVIN_CODE" \
  --output "$CALVIN_DATA/task_ABC_D_volume_g_subset/surface_g_calibration.json"

"$CALVIN_PYTHON" \
  "$AUTORESEARCH_ROOT/framework/scripts/data/build_calvin_interaction_volume_cache.py" \
  --data-root "$CALVIN_DATA/task_ABC_D_volume_g_subset" \
  --calvin-code "$CALVIN_CODE" \
  --calibration-report \
    "$CALVIN_DATA/task_ABC_D_volume_g_subset/surface_g_calibration.json"
```

The calibration must report `heldout_used_for_fitting: false`. The expected
source-only cache has 96 segments and 5,769 frames, balanced at 24 per source
task and 32 per scene. It stores `robot_obs[5769,15]`,
`scene_obs[5769,24]`, `rel_actions[5769,7]`, and `minimal_g[5769,6]`; the
audited compressed size is 1,247,384 bytes. Held-out task frames remain in the
raw subset only for frozen representation audit and are never written into the
training cache.

The cache label is
`G=[delta_to_expanded_oriented_object_volume(3), desired_object_flow(3)]`.
Attraction becomes sticky zero after 2 mm directed object progress while flow
remains active. This is a matched relabel-and-retrain gate; it must not be
inserted into a checkpoint trained with contact-anchor G.

### Frozen DINOv2 dual-camera features

Model source:

- Hugging Face model: <https://huggingface.co/facebook/dinov2-small>
- Model id: `facebook/dinov2-small`
- Snapshot used on the audited server:
  `ed25f3a31f01632728cabb09d1542f84ab7b0056`
- Runtime: PyTorch `2.3.1+cu121`, Transformers `4.46.3`

Generate the cache:

```bash
CUDA_VISIBLE_DEVICES=0 "$CALVIN_PYTHON" \
  "$AUTORESEARCH_ROOT/framework/scripts/data/precompute_calvin_dinov2_features_from_zip.py" \
  --source-zip "$CALVIN_DATA/task_ABC_D.zip" \
  --compact-data-root "$CALVIN_DATA/task_ABC_D_compact_blocks" \
  --output-dir "$CALVIN_DATA/task_ABC_D_compact_blocks_dinov2_small" \
  --model-name facebook/dinov2-small \
  --batch-size 128 \
  --shard-size 2048 \
  --device cuda:0
```

For strict recovery, pre-download the recorded Hugging Face snapshot and keep
it in the machine's Hugging Face cache. The current script records the model id
in `manifest.json`; the snapshot hash above closes the historical provenance
gap for the cache already used.

Expected cache invariants:

| Field | Expected value |
| --- | --- |
| frames | 51,594 |
| cameras | `rgb_static`, `rgb_gripper` |
| feature dimension | 384 per camera |
| feature | DINOv2 CLS token |
| dtype | float16 |
| image size | 224 x 224 |
| shard size | 2,048 |
| disk use | approximately 77 MB |

Online evaluation loads the same frozen model and ImageNet normalization. It
does not train DINOv2.

## Source 2: project-collected Isaac Lab demonstrations

Phase-0 lift data is synthetic data generated inside Isaac Lab. There is no
external dataset download. The teacher, task modifications, camera setup, and
collector are vendored in this repository under `simulators/isaaclab/`.

Canonical collection command:

```bash
export AUTORESEARCH_ROOT="$PWD"
source simulators/resolve_paths.sh

STAGE=s1_random_cube \
RANDOMIZE_CUBE=1 \
CLEAN_TEACHER=1 \
CAMERA_SIZE=240 \
TOPDOWN_CLOSE=1 \
NUM_DEMOS=100 \
bash scripts/collect_lift_demos.sh
```

The collection path is always project-owned:

```text
data/lift/<stage>/demos/*.hdf5
```

Never train directly from or write new data into
`$ISAACLAB_ROOT/datasets/`. To import an older, read-only shard:

```bash
python scripts/import_isaac_demos.py \
  --source "$ISAACLAB_ROOT/datasets/<dataset>.hdf5" \
  --task lift \
  --stage s0_fixed
```

Committed manifests under `data/manifests/` record the original absolute path,
collection options, and where available the SHA-256. The two fixed-cube source
shards previously used were:

```text
2632cffd05f2a17861fe0de1cf8e5a33f72e329886b5a3aea9317998d9dcb284  lift_cube_lenient_teacher_maskdepth_smoke.hdf5
2e2f581c000a503315f49fa404e377be59c00c5baaa772e86cda9b5caa867cd7  lift_cube_lenient_teacher_maskdepth_v3.hdf5
```

The raw HDF5 files are not required to continue the active CALVIN phase. If a
bit-identical Phase-0 archive is unavailable, regenerate the stage with the
same manifest options and treat it as a new collection seed rather than
claiming byte identity.

## Integrity checks

After recovery:

```bash
sha256sum "$CALVIN_DATA/task_ABC_D.zip"

"$CALVIN_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

data = Path(os.environ["CALVIN_DATA"])
compact = json.loads((data / "task_ABC_D_compact_blocks/compact_extract_manifest.json").read_text())
visual = json.loads((data / "task_ABC_D_compact_blocks_dinov2_small/manifest.json").read_text())
assert compact["complete"] and compact["segments"] == 2101
assert compact["episode_files_total"] == 51594 and not compact["missing_episode_ids"]
assert visual["num_frames"] == 51594 and visual["feature_dim"] == 384
assert visual["cameras"] == ["rgb_static", "rgb_gripper"]
print("dataset manifests: OK")
PY
```

## What Git stores

Git stores collection scripts, manifests, environment freezes, run metadata,
metrics, and lightweight JSON/CSV evidence. It intentionally excludes zips,
HDF5, NPZ/NPY arrays, model weights, videos, caches, and simulator installs.
See `bootstrap.md` for the full empty-machine recovery order.
